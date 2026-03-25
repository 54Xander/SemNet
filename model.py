# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, AttentionalAggregation

# --- 基础模块 (保持不变) ---

class PretrainedBranch(nn.Module):
    """
    轻量级映射层：仅用于将离线提取的特征映射到模型维度。
    """
    def __init__(self, input_dim=768, embed_dim=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        return self.proj(x)

class RNA_RGCN_Advanced(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args['embed_dim']
        self.num_relations = 9
        
        self.conv1 = RGCNConv(5, self.dim, num_relations=self.num_relations)
        self.conv2 = RGCNConv(self.dim, self.dim, num_relations=self.num_relations)
        
        self.gate_nn = nn.Linear(self.dim, 1)
        self.pool = AttentionalAggregation(gate_nn=self.gate_nn)
        self.fc = nn.Linear(self.dim, self.dim)

    def forward(self, x, edge_index, edge_type, batch):
        x = x.float().contiguous() 
        edge_index = edge_index.long()
        edge_type = edge_type.long()

        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)

        out = self.pool(x, batch)
        return self.fc(out)

class DynamicWeightNet(nn.Module):
    def __init__(self, input_dim, num_modalities):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_modalities)
        
    def forward(self, x, mask=None):
        logits = self.fc(x) 
        if mask is not None:
            mask = mask.view(logits.size(0), -1)
            logits = logits.masked_fill(mask == 0, -1e9)
        return F.softmax(logits, dim=1)
    
    
# --- 主模型 (修正了 AttributeError) ---

class MultiModalModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args['embed_dim']
        
        # 1. 获取自适应维度
        chem_dim = args.get('mol_input_dim', 384) 
        bio_dim  = args.get('bio_input_dim', 768) 
        rna_dim  = args.get('rna_input_dim', 120) 
        
        print(f"[Model] 自适应初始化: Chem={chem_dim}, Bio={bio_dim}, RNA={rna_dim}")

        # ========================================================
        # 【关键修复】 显式将开关保存到 self，供 forward 使用
        # ========================================================
        self.use_rna_2d = args.get('use_rna_2d', 1)
        self.use_mol_sem = args.get('use_mol_sem', 1)
        self.use_rna_sem = args.get('use_rna_sem', 1)
        # ========================================================

        # --- 小分子分支 ---
        self.mol_chem = PretrainedBranch(input_dim=chem_dim, embed_dim=self.embed_dim)
        mol_mod_count = 1
        
        # 使用 self.use_mol_sem 判断
        if self.use_mol_sem:
            self.mol_bio = PretrainedBranch(input_dim=bio_dim, embed_dim=self.embed_dim)
            mol_mod_count += 1
        self.mol_weight_net = DynamicWeightNet(mol_mod_count * self.embed_dim, mol_mod_count)
        
        # --- RNA 分支 ---
        self.rnabert = PretrainedBranch(input_dim=rna_dim, embed_dim=self.embed_dim)
        rna_mod_count = 1
        
        if self.use_rna_sem:
            self.rna_bio = PretrainedBranch(input_dim=bio_dim, embed_dim=self.embed_dim)
            rna_mod_count += 1
            
        if self.use_rna_2d:
            self.rna_rgcn = RNA_RGCN_Advanced(args)
            rna_mod_count += 1
        self.rna_weight_net = DynamicWeightNet(rna_mod_count * self.embed_dim, rna_mod_count)

        # --- 交互与输出 ---
        self.MHA_m_from_r = nn.MultiheadAttention(self.embed_dim, args['nhead'], batch_first=True)
        self.MHA_r_from_m = nn.MultiheadAttention(self.embed_dim, args['nhead'], batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(args['dropout']),
            nn.Linear(self.embed_dim, args['n_output'])
        )

    def forward(self, data):
        bs = data.y.size(0)
        
        # --- 小分子处理 ---
        m_list = [self.mol_chem(data.mol_chem_feat.view(bs, -1))]
        m_mask_indices = [0]
        
        # 这里的 self.use_mol_sem 现在已经定义了，不会再报错
        if self.use_mol_sem:
            m_list.append(self.mol_bio(data.mol_sem_feat.view(bs, -1)))
            m_mask_indices.append(1)
        
        m_mask = getattr(data, 'mol_mask', None)
        if m_mask is not None:
            # 确保 mask 维度与当前使用的模态数量对齐
            # 这里的切片 [:len(m_mask_indices)] 是一种更安全的写法
            m_mask = m_mask.view(bs, -1)[:, m_mask_indices]
            
        mw = self.mol_weight_net(torch.cat(m_list, dim=1), mask=m_mask).unsqueeze(2)
        mol_feat = torch.sum(torch.stack(m_list, dim=1) * mw, dim=1)

        # --- RNA 处理 ---
        r_list = [self.rnabert(data.rna_bert_feat.view(bs, -1))]
        r_mask_indices = [0]
        
        if self.use_rna_sem:
            r_list.append(self.rna_bio(data.rna_sem_feat.view(bs, -1)))
            r_mask_indices.append(1)
            
        if self.use_rna_2d:
            r_list.append(self.rna_rgcn(data.rna_2d_x, data.rna_2d_edge_index, data.rna_2d_edge_type, data.batch))
            r_mask_indices.append(2)
        
        r_mask = getattr(data, 'rna_mask', None)
        if r_mask is not None:
            r_mask = r_mask.view(bs, -1)[:, r_mask_indices]
            
        rw = self.rna_weight_net(torch.cat(r_list, dim=1), mask=r_mask).unsqueeze(2)
        rna_feat = torch.sum(torch.stack(r_list, dim=1) * rw, dim=1)

        # --- 交互与输出 ---
        m_ctx = self.MHA_m_from_r(mol_feat.unsqueeze(1), rna_feat.unsqueeze(1), rna_feat.unsqueeze(1))[0].squeeze(1)
        r_ctx = self.MHA_r_from_m(rna_feat.unsqueeze(1), mol_feat.unsqueeze(1), mol_feat.unsqueeze(1))[0].squeeze(1)
        out = self.classifier(torch.cat([m_ctx, r_ctx], dim=1))
        
        return {
            'out': out, 
            'drug_feat': m_ctx, 
            'rna_feat': r_ctx,
            'mol_weights': mw.squeeze(),
            'rna_weights': rw.squeeze()
        }

class InteractionContrastiveLoss(nn.Module):
    """ 保持不变 """
    def __init__(self, temperature=0.1):
        super(InteractionContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, rna_emb, drug_emb, labels):
        rna_emb = F.normalize(rna_emb, p=2, dim=1)
        drug_emb = F.normalize(drug_emb, p=2, dim=1)
        logits = torch.matmul(rna_emb, drug_emb.T) / self.temperature
        pos_mask = (labels == 1).float()
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=rna_emb.device, requires_grad=True)
        batch_size = rna_emb.size(0)
        target = torch.arange(batch_size).to(rna_emb.device)
        loss_rna = F.cross_entropy(logits, target, reduction='none')
        loss_drug = F.cross_entropy(logits.T, target, reduction='none')
        interaction_loss = ((loss_rna + loss_drug) / 2) * pos_mask
        return interaction_loss.sum() / (pos_mask.sum() + 1e-7)