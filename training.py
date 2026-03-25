# training.py
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from utils import TestbedDataset, set_seed, get_metrics
import random
import gc  # 引入垃圾回收模块
from model import MultiModalModel, InteractionContrastiveLoss

# 禁用 HF 链接警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# 限制 PyTorch 显存碎片，有助于减少 OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 尝试导入数据生成脚本
try:
    import create_data
except ImportError:
    print("Warning: create_data.py not found. Auto-generation disabled.")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ==========================
# 1. 数据加载与生成逻辑
# ==========================

def check_data_exists(args):
    """
    检查数据是否存在，不存在则自动生成
    """
    dataset_name = args['dataset']
    check_path = f'data/processed/{dataset_name}_fold1_tra_{args["split_mode"]}.pt'
    
    if not os.path.exists(check_path):
        print(f"\n[Auto-Check] 未检测到数据文件: {check_path}")
        print(f"[Auto-Check] 正在调用 create_data.py 自动生成 (Split: {args['split_mode']})...")
        try:
            create_data.run_generation(args)
        except NameError:
            raise FileNotFoundError("自动生成失败：找不到 create_data 模块。")
        except Exception as e:
            raise RuntimeError(f"自动生成数据时发生错误: {str(e)}")
    else:
        print(f"[Auto-Check] 检测到离线数据 ({args['split_mode']})，跳过生成。")

def load_single_fold(args, fold):
    """
    只加载当前这一折的数据到内存，节省 RAM 和显存
    """
    dataset_name = args['dataset']
    print(f"[Load] 正在加载 Fold {fold} 离线数据...")
    
    # 1. 直接读取 .pt 文件
    tra_list = torch.load(f'data/processed/{dataset_name}_fold{fold}_tra_{args["split_mode"]}.pt', weights_only=False)
    val_list = torch.load(f'data/processed/{dataset_name}_fold{fold}_val_{args["split_mode"]}.pt', weights_only=False)
    tes_list = torch.load(f'data/processed/{dataset_name}_fold{fold}_tes_{args["split_mode"]}.pt', weights_only=False)

    # 2. 包装进 Dataset
    tra_ds = TestbedDataset(root='data', dataset='tmp'); tra_ds.data, tra_ds.slices = tra_ds.collate(tra_list)
    val_ds = TestbedDataset(root='data', dataset='tmp'); val_ds.data, val_ds.slices = val_ds.collate(val_list)
    tes_ds = TestbedDataset(root='data', dataset='tmp'); tes_ds.data, tes_ds.slices = tes_ds.collate(tes_list)

    # 3. 创建 DataLoader
    g = torch.Generator()
    g.manual_seed(args['seed'])
    
    loader_tra = DataLoader(tra_ds, batch_size=args['batch_size'], shuffle=True, 
                            num_workers=args['num_workers'], pin_memory=args['pin_memory'], 
                            worker_init_fn=seed_worker, generator=g)
    loader_val = DataLoader(val_ds, batch_size=args['batch_size'], shuffle=False, 
                            num_workers=args['num_workers'], pin_memory=args['pin_memory'], 
                            worker_init_fn=seed_worker, generator=g)
    loader_tes = DataLoader(tes_ds, batch_size=args['batch_size'], shuffle=False, 
                            num_workers=args['num_workers'], pin_memory=args['pin_memory'], 
                            worker_init_fn=seed_worker, generator=g)
    
    return loader_tra, loader_val, loader_tes

def compute_pos_weight(loader):
    labels = [data.y.item() for data in loader.dataset]
    labels = np.array(labels)
    num_neg = (labels == 0).sum()
    num_pos = (labels == 1).sum()
    return torch.tensor([(num_neg + 1e-7) / (num_pos + 1e-7)], dtype=torch.float32)

# ==========================
# 2. 训练与评估核心步
# ==========================

def train_one_epoch(model, loader, loss_fn, optimizer, device, args):
    model.train()
    total_loss, y_true, y_pred = 0, [], []
    interaction_contra_loss_fn = InteractionContrastiveLoss(temperature=args['contrastive_temp'])

    for data in loader:
        data = data.to(device)
        target = data.y.view(-1, 1).float().to(device)
        optimizer.zero_grad()
        output = model(data)
        logits = output['out']
        bce_loss = loss_fn(logits, target)
        
        rna_emb = output['rna_feat']
        drug_emb = output['drug_feat']
        interaction_loss = interaction_contra_loss_fn(rna_emb, drug_emb, data.y.view(-1))
        
        loss = bce_loss + args['aux_weight_interaction'] * interaction_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * target.size(0)
        y_pred += torch.sigmoid(logits).detach().cpu().view(-1).tolist()
        y_true += data.y.view(-1).cpu().tolist()

    return round(total_loss / len(y_true), 5), get_metrics(y_true, y_pred)

def evaluate(model, loader, loss_fn, device, args):
    model.eval()
    total_loss, y_true, y_pred = 0, [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            target = data.y.view(-1, 1).float().to(device)
            output = model(data)
            logits = output['out']
            loss = loss_fn(logits, target)
            total_loss += loss.item() * target.size(0)
            y_pred += torch.sigmoid(logits).cpu().view(-1).tolist()
            y_true += data.y.view(-1).cpu().tolist()
    return round(total_loss / len(y_true), 5), get_metrics(y_true, y_pred)

# ==========================
# 3. 单折运行封装
# ==========================

def run_single_fold(fold, args, device):
    """
    单折训练核心封装
    """
    # 1. 加载当前折数据
    loader_tra, loader_val, loader_tes = load_single_fold(args, fold)
    
    # 2. 自动检测维度 (仅在第一折时输出信息，但每折都重新确认)
    sample_data = loader_tra.dataset[0]
    if hasattr(sample_data, 'mol_chem_feat'):
        args['mol_input_dim'] = sample_data.mol_chem_feat.shape[-1]
    if hasattr(sample_data, 'rna_bert_feat'):
        args['rna_input_dim'] = sample_data.rna_bert_feat.shape[-1]
    
    if hasattr(sample_data, 'mol_sem_feat') and sample_data.mol_sem_feat.shape[-1] > 0:
        args['bio_input_dim'] = sample_data.mol_sem_feat.shape[-1]
    elif hasattr(sample_data, 'rna_sem_feat') and sample_data.rna_sem_feat.shape[-1] > 0:
        args['bio_input_dim'] = sample_data.rna_sem_feat.shape[-1]
    else:
        args['bio_input_dim'] = 768

    print(f"\n===== Fold {fold} / {args['n_splits']} =====")
    
    # 3. 初始化模型相关
    set_seed(args['seed'])
    model = MultiModalModel(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args['scheduler_patience'], factor=args['scheduler_factor'])
    pos_weight = compute_pos_weight(loader_tra).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    dataset_name = args['dataset']
    best_auc = 0
    best_epoch = -1            
    best_tes_metrics = None    
    result_metrics = np.zeros((args['epochs'], 1 + 7 * 3))
    early_stop_counter = 0

    # 4. 训练循环
    for epoch in range(args['epochs']):
        t0 = time.time()
        loss_tra, met_tra = train_one_epoch(model, loader_tra, loss_fn, optimizer, device, args)
        loss_val, met_val = evaluate(model, loader_val, loss_fn, device, args)
        loss_tes, met_tes = evaluate(model, loader_tes, loss_fn, device, args)
        scheduler.step(loss_val)
        
        t1 = round((time.time() - t0) / 60, 2)
        print(f"Ep {epoch:03d} | Time {t1}m")
        print(f"Tra Loss: {loss_tra:.5f} | Metrics: {met_tra}")
        print(f"Val Loss: {loss_val:.5f} | Metrics: {met_val}")
        print(f"Tes Loss: {loss_tes:.5f} | Metrics: {met_tes}", flush=True)

        result_metrics[epoch, 0] = epoch
        for i in range(7):
            result_metrics[epoch, 3 * i + 1] = met_tra[i]
            result_metrics[epoch, 3 * i + 2] = met_val[i]
            result_metrics[epoch, 3 * i + 3] = met_tes[i]

        if met_val[0] > best_auc:
            best_auc = met_val[0]
            best_epoch = epoch            
            best_tes_metrics = met_tes    
            early_stop_counter = 0
            os.makedirs("model", exist_ok=True)
            torch.save(model.state_dict(), f"model/{dataset_name}_{args['split_mode']}_fold{fold}.pt")
            print(f">>> Best Tes updated at epoch {epoch:03d}: AUC={met_tes[0]}")
        else:
            early_stop_counter += 1
            print(f"!! No improvement for {early_stop_counter}/{args['early_stop_patience']} epochs")

        if early_stop_counter >= args['early_stop_patience']:
            print("Early stopping triggered.")
            break

    # 5. 保存结果
    print(f"\nFold {fold} 最佳 Epoch: {best_epoch:03d} | 最佳指标: {best_tes_metrics}")
    
    actual_epochs = epoch + 1
    result_metrics_truncated = result_metrics[:actual_epochs, :]
    cols = ['Epoch'] + [f"{p}_{m}" for m in ['AUC','AUPR','F1','Acc','Rec','Spec','Prec'] for p in ['Tra','Val','Tes']]
    df = pd.DataFrame(result_metrics_truncated, columns=cols)
    os.makedirs('result', exist_ok=True)
    time_str = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    df.to_csv(f"result/result_{dataset_name}_{args['split_mode']}_fold{fold}_{time_str}.csv", index=False)
    
    return

# ==========================
# 4. 主调度函数
# ==========================

def TraValTes(args):
    device = args['device'] 
    
    # 步骤 1: 确保数据已生成
    try:
        check_data_exists(args)
    except Exception as e:
        print(f"数据准备失败: {e}")
        return

    # 步骤 2: 循环训练折
    for fold in range(1, args['n_splits'] + 1):
        
        # 运行单折训练 (利用局部作用域释放大部分内存)
        run_single_fold(fold, args, device)

        # 步骤 3: 每一折结束后的硬核清理
        print(f"正在深度清理 Fold {fold} 显存...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(2)
        print(f"Fold {fold} 清理完毕。\n")

# ==========================
# 5. 参数解析
# ==========================

def get_args():
    parser = argparse.ArgumentParser()
    # 基础
    parser.add_argument('--dataset', default='RSID')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32) 
    
    # 训练
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--split_mode', type=str, default='random') 
    
    #  'random', 'cold_rna', 'cold_drug', 'cold_both'
    
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--n_output', type=int, default=1)
    
    # 损失
    parser.add_argument('--contrastive_temp', type=float, default=0.3)
    parser.add_argument('--aux_weight_interaction', type=float, default=0.3)
    
    # 调度
    parser.add_argument('--scheduler_patience', type=int, default=7)
    parser.add_argument('--scheduler_factor', type=float, default=0.7)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    
    # 硬件
    parser.add_argument('--pin_memory', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0) 
    
    # 开关
    parser.add_argument('--use_rna_2d', type=int, default=1)
    parser.add_argument('--use_mol_sem', type=int, default=1)
    parser.add_argument('--use_rna_sem', type=int, default=1)
    
    # 路径
    parser.add_argument('--mol_chem_path', type=str, default='./pretrained_models/ChemBERTa-77M-MTR')
    parser.add_argument('--mol_sem_path', type=str, default='./pretrained_models/biobert-base-cased-v1.2')
    parser.add_argument('--rnabert_path', type=str, default='./pretrained_models/rnabert')
    parser.add_argument('--max_seq_len', type=int, default=267)  # 220
    parser.add_argument('--max_mol_len', type=int, default=241)
    parser.add_argument('--val_size', type=float, default=0.1)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args_dict = vars(args)
    args_dict['device'] = device
    return args_dict

if __name__ == '__main__':
    args_config = get_args()
    set_seed(args_config['seed'])
    TraValTes(args_config)