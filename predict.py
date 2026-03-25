import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, AutoModel
from model import MultiModalModel
from create_data import SimpleRnaTokenizer, extract_feature
from utils import rna2D_from_dot

# --- 1. 配置与参数设置 ---
args = {
    'embed_dim': 512,
    'nhead': 8,
    'dropout': 0.3,
    'n_output': 1,
    'use_rna_2d': 1,
    'use_mol_sem': 1,
    'use_rna_sem': 1,
    'max_seq_len': 220,
    'max_mol_len': 241,
    'mol_chem_path': './pretrained_models/ChemBERTa-77M-MTR',
    'mol_sem_path': './pretrained_models/biobert-base-cased-v1.2',
    'rnabert_path': './pretrained_models/rnabert',
    'model_weight': 'model/RSID_random_fold1.pt' # 请确保此路径指向你训练最好的模型
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_feature_extractors():
    print("[Init] 正在加载特征提取模型 (这可能需要一点时间)...")
    # Tokenizers
    tok_chem = AutoTokenizer.from_pretrained(args['mol_chem_path'], local_files_only=True)
    tok_bio = AutoTokenizer.from_pretrained(args['mol_sem_path'], local_files_only=True)
    tok_rna = SimpleRnaTokenizer.from_pretrained(args['rnabert_path'])
    
    # Models
    m_chem = AutoModel.from_pretrained(args['mol_chem_path'], local_files_only=True).to(device).eval()
    m_bio = AutoModel.from_pretrained(args['mol_sem_path'], local_files_only=True).to(device).eval()
    m_rna = AutoModel.from_pretrained(args['rnabert_path'], local_files_only=True, trust_remote_code=True).to(device).eval()
    
    return (tok_chem, tok_bio, tok_rna), (m_chem, m_bio, m_rna)

def process_row(row, tokenizers, models):
    """将单行 Excel 数据转换为模型所需的 Data 对象"""
    t_chem, t_bio, t_rna = tokenizers
    m_chem, m_bio, m_rna = models
    
    smiles = str(row['SMILES']) if pd.notna(row['SMILES']) else ""
    mol_text = str(row['Small molecule information']) if pd.notna(row['Small molecule information']) else ""
    rna_seq = str(row['1D Sequence']) if pd.notna(row['1D Sequence']) else ""
    rna_text = str(row['RNA information']) if pd.notna(row['RNA information']) else ""
    dot_bracket = str(row['Dot bracket']) if pd.notna(row['Dot bracket']) else ""

    with torch.no_grad():
        # 小分子特征
        mc_in = t_chem(smiles, max_length=args['max_mol_len'], padding='max_length', truncation=True, return_tensors='pt')
        mol_chem_feat = extract_feature(m_chem, mc_in, device)
        
        ms_in = t_bio(f"{smiles} [SEP] {mol_text}", max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        mol_sem_feat = extract_feature(m_bio, ms_in, device)
        
        # RNA 特征
        rf_in = t_rna(rna_seq, max_length=args['max_seq_len'], padding='max_length', truncation=True, return_tensors='pt')
        rna_bert_feat = extract_feature(m_rna, rf_in, device)
        
        rs_in = t_bio(f"{rna_seq} [SEP] {rna_text}", max_length=args['max_seq_len'], padding='max_length', truncation=True, return_tensors='pt')
        rna_sem_feat = extract_feature(m_bio, rs_in, device)

        # 2D 结构
        res_2d = rna2D_from_dot(rna_seq[:args['max_seq_len']].ljust(args['max_seq_len'], 'N'), 
                                 dot_bracket[:args['max_seq_len']].ljust(args['max_seq_len'], '.'))
        rna_2d_x, rna_2d_edge_index, rna_2d_edge_type = res_2d

    # 构建 Mask (逻辑同 training.py)
    mol_mask = torch.tensor([1 if len(smiles) > 5 else 0, 1 if len(mol_text) > 5 else 0], dtype=torch.float)
    rna_mask = torch.tensor([1 if len(rna_seq) > 0 else 0, 1 if len(rna_text) > 5 else 0, 1 if "(" in dot_bracket else 0], dtype=torch.float)

    from torch_geometric.data import Data
    return Data(
        mol_chem_feat=mol_chem_feat.squeeze(0),
        mol_sem_feat=mol_sem_feat.squeeze(0),
        rna_bert_feat=rna_bert_feat.squeeze(0),
        rna_sem_feat=rna_sem_feat.squeeze(0),
        rna_2d_x=rna_2d_x.float(),
        rna_2d_edge_index=rna_2d_edge_index.long(),
        rna_2d_edge_type=rna_2d_edge_type.long(),
        mol_mask=mol_mask,
        rna_mask=rna_mask,
        y=torch.tensor([0.0]) # 占位符
    )

def main():
    # 1. 准备路径
    input_path = 'prediction/example.xlsx'
    output_path = 'prediction/prediction_results.xlsx'
    if not os.path.exists('prediction'): os.makedirs('prediction')

    # 2. 加载数据
    df = pd.read_excel(input_path)
    print(f"[Data] 加载预测任务，共 {len(df)} 条数据。")

    # 3. 初始化特征提取器与主模型
    tokenizers, extractors = load_feature_extractors()
    
    # 动态获取维度信息
    args['mol_input_dim'] = 384 # ChemBERTa 默认
    args['rna_input_dim'] = 120 # RNABERT 默认
    args['bio_input_dim'] = 768 # BioBERT 默认
    
    model = MultiModalModel(args).to(device)
    model.load_state_dict(torch.load(args['model_weight'], map_location=device))
    model.eval()

    # 4. 执行预测
    results = []
    print("[Predict] 正在推理...")
    for idx, row in df.iterrows():
        data_obj = process_row(row, tokenizers, extractors).to(device)
        # 为 Data 对象添加 batch 属性 (RGCN 必需)
        data_obj.batch = torch.zeros(data_obj.rna_2d_x.size(0), dtype=torch.long).to(device)
        
        with torch.no_grad():
            output = model(data_obj)
            prob = torch.sigmoid(output['out']).item()
            results.append(prob)
        
        if (idx + 1) % 10 == 0: print(f"  已完成: {idx + 1}/{len(df)}")

    # 5. 保存结果
    df['Interaction_Probability'] = results
    df['Prediction'] = (df['Interaction_Probability'] >= 0.5).astype(int)
    df.to_excel(output_path, index=False)
    print(f"\n[Done] 预测结果已保存至: {output_path}")

if __name__ == "__main__":
    main()