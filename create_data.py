import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import RDLogger
import torch
import torch.nn as nn
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel, BertModel
from torch_geometric.data import Batch
import argparse

RDLogger.DisableLog('rdApp.*')
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from utils import (
    TestbedDataset,
    rna2D_from_dot
)

class SimpleRnaTokenizer:
    """Manually implemented tokenizer compatible with the RNABERT vocabulary."""
    def __init__(self, model_path):
        vocab_path = os.path.join(model_path, "vocab.txt")
        self.vocab = {}
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    self.vocab[line.strip()] = i
        else:
            self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "A": 4, "U": 5, "G": 6, "C": 7, "N": 8}
        
        self.pad_token_id = self.vocab.get("[PAD]", 0)
        self.cls_token_id = self.vocab.get("[CLS]", 2)
        self.sep_token_id = self.vocab.get("[SEP]", 3)
        self.unk_token_id = self.vocab.get("[UNK]", 1)

    def __call__(self, text, max_length=220, padding='max_length', truncation=True, return_tensors='pt'):
        text = str(text) if text is not None else ""
        tokens = list(text.upper())
        input_ids = [self.vocab.get(t, self.unk_token_id) for t in tokens]
        if truncation and len(input_ids) > (max_length - 2):
            input_ids = input_ids[:max_length - 2]
        input_ids = [self.cls_token_id] + input_ids + [self.sep_token_id]
        attention_mask = [1] * len(input_ids)
        if padding == 'max_length' and len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            input_ids += [self.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            
        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor([input_ids]),
                'attention_mask': torch.tensor([attention_mask])
            }
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    @classmethod
    def from_pretrained(cls, path):
        return cls(path)

def extract_feature(model, inputs, device):
    """Generic feature extraction logic: input token IDs, output CLS vector."""
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            feat = outputs.pooler_output
        else:
            feat = outputs.last_hidden_state[:, 0, :]
    return feat.cpu()

def read_raw_data(dataset_path, n_splits=5, seed=42, val_size=0.1, split_mode='random'):
    """ 
    Perform data splitting based on different modes:
    - random: random sample-level split
    - cold_rna: split by RNA ID (new RNA prediction)
    - cold_drug: split by molecule ID (new molecule prediction)
    - cold_both: strict dual cold-start (new RNA + new molecule)
    """
    dataset_name = os.path.basename(dataset_path)
    csv_dir = 'data/processed'
    os.makedirs(csv_dir, exist_ok=True)
    
    df_molecules = pd.read_excel(os.path.join(dataset_path, "Molecule.xlsx"))
    molecules_map = {row['Small molecule_ID']: row['SMILES'] for _, row in df_molecules.iterrows()}
    mol_info_map = {row['Small molecule_ID']: str(row['Small molecule information']) if pd.notna(row['Small molecule information']) else "" 
                    for _, row in df_molecules.iterrows()}

    df_rnas = pd.read_excel(os.path.join(dataset_path, "RNA.xlsx")).set_index('RNA_ID')
    if 'RNA information' not in df_rnas.columns:
        df_rnas['RNA information'] = ""
    
    df_labels = pd.read_excel(os.path.join(dataset_path, "RNA-Molecule.xlsx"))
    
    mol_id_col = 'Small molecule_ID'
    label_col = 'label'
    unique_rnas = df_labels['RNA_ID'].unique()
    unique_mols = df_labels[mol_id_col].unique()

    from sklearn.model_selection import KFold, StratifiedKFold
    np.random.seed(seed)
    
    if split_mode == 'random':
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(kf.split(df_labels, df_labels[label_col]))
    elif split_mode == 'cold_rna':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(kf.split(unique_rnas))
    elif split_mode == 'cold_drug':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(kf.split(unique_mols))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        rna_splits = list(kf.split(unique_rnas))
        mol_splits = list(kf.split(unique_mols))

    all_folds_data = []

    for fold_idx in range(n_splits):
        fold = fold_idx + 1
        
        if split_mode == 'random':
            train_idx, test_idx = splits[fold_idx]
            df_train_full = df_labels.iloc[train_idx]
            df_test = df_labels.iloc[test_idx]
            df_train, df_val = train_test_split(
                df_train_full, test_size=val_size, stratify=df_train_full[label_col], random_state=seed
            )
            
        elif split_mode == 'cold_rna':
            train_val_idx, test_idx = splits[fold_idx]
            train_val_rnas = unique_rnas[train_val_idx]
            test_rnas = unique_rnas[test_idx]
            tra_rnas, val_rnas = train_test_split(train_val_rnas, test_size=val_size, random_state=seed)
            df_train = df_labels[df_labels['RNA_ID'].isin(tra_rnas)]
            df_val = df_labels[df_labels['RNA_ID'].isin(val_rnas)]
            df_test = df_labels[df_labels['RNA_ID'].isin(test_rnas)]
            
        elif split_mode == 'cold_drug':
            train_val_idx, test_idx = splits[fold_idx]
            train_val_mols = unique_mols[train_val_idx]
            test_mols = unique_mols[test_idx]
            tra_mols, val_mols = train_test_split(train_val_mols, test_size=val_size, random_state=seed)
            df_train = df_labels[df_labels[mol_id_col].isin(tra_mols)]
            df_val = df_labels[df_labels[mol_id_col].isin(val_mols)]
            df_test = df_labels[df_labels[mol_id_col].isin(test_mols)]

        elif split_mode == 'cold_both':
            (rna_tra_val_idx, rna_tes_idx) = rna_splits[fold_idx]
            (mol_tra_val_idx, mol_tes_idx) = mol_splits[fold_idx]
            
            test_rnas, test_mols = unique_rnas[rna_tes_idx], unique_mols[mol_tes_idx]
            tv_rnas, tv_mols = unique_rnas[rna_tra_val_idx], unique_mols[mol_tra_val_idx]
            
            tra_rnas, val_rnas = train_test_split(tv_rnas, test_size=val_size, random_state=seed)
            tra_mols, val_mols = train_test_split(tv_mols, test_size=val_size, random_state=seed)
            
            df_train = df_labels[df_labels['RNA_ID'].isin(tra_rnas) & df_labels[mol_id_col].isin(tra_mols)]
            df_val = df_labels[df_labels['RNA_ID'].isin(val_rnas) & df_labels[mol_id_col].isin(val_mols)]
            df_test = df_labels[df_labels['RNA_ID'].isin(test_rnas) & df_labels[mol_id_col].isin(test_mols)]

        processed_dfs = []
        for df_sub, tvt_type in zip([df_train, df_val, df_test], ['tra', 'val', 'tes']):
            df_sub = df_sub.merge(df_rnas, left_on='RNA_ID', right_index=True)
            df_sub['SMILES'] = df_sub[mol_id_col].map(molecules_map)
            df_sub['Small_molecule_information'] = df_sub[mol_id_col].map(mol_info_map)
            
            df_sub = df_sub.dropna(subset=['SMILES'])
            
            out_path = os.path.join(csv_dir, f'{dataset_name}_fold{fold}_{tvt_type}_{split_mode}.csv')
            df_sub.to_csv(out_path, index=False)
            processed_dfs.append(df_sub)
            
        all_folds_data.append(tuple(processed_dfs))
        print(f"  [Fold {fold}] Split complete. Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    return all_folds_data

def trans_multimodal(dataset_path, df_data, tvt_type, fold, args, models_dict, device):
    """
    Updated processing function: perform offline feature extraction.
    """
    dataset_name = os.path.basename(dataset_path)
    split_mode = args['split_mode'] if isinstance(args, dict) else args.split_mode
    pt_path = f'data/processed/{dataset_name}_fold{fold}_{tvt_type}_{split_mode}.pt'

    if os.path.exists(pt_path):
        print(f"[Cache] Loading offline feature data: {pt_path}")
        data_list = torch.load(pt_path)
        return data_list

    print(f'[Process] Generating offline features for {tvt_type} (Fold {fold})...')

    def get_arg(key): return args[key] if isinstance(args, dict) else getattr(args, key)

    tok_chem = AutoTokenizer.from_pretrained(get_arg('mol_chem_path'), local_files_only=True)
    tok_bio = AutoTokenizer.from_pretrained(get_arg('mol_sem_path'), local_files_only=True)
    tok_fm = SimpleRnaTokenizer.from_pretrained(get_arg('rnabert_path'))
    max_len = get_arg('max_seq_len')

    model_chem = models_dict['mol_chem']
    model_bio = models_dict['mol_sem'] 
    model_rna = models_dict['rnabert']

    data_list = []
    
    for idx, (original_index, row) in enumerate(df_data.iterrows()):
        if idx % 100 == 0: 
            print(f"  Processing {idx}/{len(df_data)} ({(idx/len(df_data))*100:.0f}%)...", end='\r')
        
        smiles = str(row['SMILES']) if pd.notna(row['SMILES']) else ""
        mol_text = str(row['Small_molecule_information']) if pd.notna(row['Small_molecule_information']) else ""
        rna_seq = str(row['1D Sequence']) if pd.notna(row['1D Sequence']) else ""
        rna_text = str(row['RNA information']) if pd.notna(row['RNA information']) else ""
        dot_bracket = str(row['Dot bracket']) if pd.notna(row['Dot bracket']) else ""

        max_mol_len = get_arg('max_mol_len')
        mc_tokens = tok_chem(smiles, max_length=max_mol_len, padding='max_length', truncation=True, return_tensors='pt')
        mol_chem_feat = extract_feature(model_chem, mc_tokens, device)

        mol_sem_feat = torch.zeros(1, 768)
        if get_arg('use_mol_sem'):
            ms_tokens = tok_bio(f"{smiles} [SEP] {mol_text}", max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            mol_sem_feat = extract_feature(model_bio, ms_tokens, device)

        mol_mask = torch.tensor([1 if len(smiles) > 5 else 0, 1 if len(mol_text) > 5 else 0], dtype=torch.float)

        rf_tokens = tok_fm(rna_seq, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
        rna_bert_feat = extract_feature(model_rna, rf_tokens, device)

        rna_sem_feat = torch.zeros(1, 768)
        if get_arg('use_rna_sem'):
            rs_tokens = tok_bio(f"{rna_seq} [SEP] {rna_text}", max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
            rna_sem_feat = extract_feature(model_bio, rs_tokens, device)

        is_valid_2d = 1 if (pd.notna(dot_bracket) and "(" in dot_bracket) else 0
        rna_2d_x = torch.zeros((max_len, 5), dtype=torch.float)
        rna_2d_edge_index = torch.empty((2, 0), dtype=torch.long)
        rna_2d_edge_type = torch.empty((0,), dtype=torch.long)
        
        if is_valid_2d:
            try:
                res_2d = rna2D_from_dot(rna_seq[:max_len].ljust(max_len, 'N'), dot_bracket[:max_len].ljust(max_len, '.'))
                if res_2d[0] is not None:
                    rna_2d_x, rna_2d_edge_index, rna_2d_edge_type = res_2d
            except Exception:
                is_valid_2d = 0

        rna_mask = torch.tensor([1 if len(rna_seq) > 0 else 0, 1 if len(rna_text) > 5 else 0, is_valid_2d], dtype=torch.float)

        data = Data(
            y=torch.tensor([row['label']], dtype=torch.float),
            mol_chem_feat=mol_chem_feat.squeeze(0),
            mol_sem_feat=mol_sem_feat.squeeze(0),
            rna_bert_feat=rna_bert_feat.squeeze(0),
            rna_sem_feat=rna_sem_feat.squeeze(0),
            rna_2d_x=rna_2d_x.float().contiguous(),
            rna_2d_edge_index=rna_2d_edge_index.long().contiguous(),
            rna_2d_edge_type=rna_2d_edge_type.long().contiguous(),
            mol_mask=mol_mask, 
            rna_mask=rna_mask,
            num_nodes=rna_2d_x.size(0)
        )
        data_list.append(data)

    os.makedirs('data/processed', exist_ok=True)
    torch.save(data_list, pt_path)
    return data_list

def run_generation(args):
    """
    Main entry for external calls.
    args: dict or Namespace, must contain the required parameters.
    """
    args_dict = vars(args) if not isinstance(args, dict) else args
    
    print("="*50)
    print(f"[Auto-Gen] Triggering automatic data generation workflow...")
    print(f"  Dataset:    {args_dict.get('dataset', 'RSID')}")
    print(f"  Split Mode: {args_dict.get('split_mode', 'random')}")
    print("="*50)

    if 'dataset_path' not in args_dict:
        args_dict['dataset_path'] = os.path.join('data', args_dict.get('dataset', 'RSID'))
    
    if not os.path.exists(args_dict['dataset_path']):
        raise FileNotFoundError(f"Dataset path does not exist: {args_dict['dataset_path']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Init] Loading pretrained models to {device} (this may take a few minutes)...")
    
    # def load_model(path):
    #     try:
    #         return BertModel.from_pretrained(path, local_files_only=True).to(device).eval()
    #     except:
    #         return AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=True).to(device).eval()
        
    def load_model(path):
        print(f"  > Loading model from {path}...")
        return AutoModel.from_pretrained(
            path, 
            local_files_only=True, 
            trust_remote_code=True
        ).to(device).eval()

    models_dict = {
        'mol_chem': load_model(args_dict['mol_chem_path']),
        'mol_sem': load_model(args_dict['mol_sem_path']), 
        'rnabert': load_model(args_dict['rnabert_path'])
    }
    
    n_splits = args_dict.get('n_splits', 5)
    seed = args_dict.get('seed', 42)
    val_size = args_dict.get('val_size', 0.1)
    split_mode = args_dict.get('split_mode', 'random')

    all_folds = read_raw_data(args_dict['dataset_path'], n_splits, seed, val_size, split_mode)
    
    for fold, (df_tra, df_val, df_tes) in enumerate(all_folds, start=1):
        for df, tvt_type in zip([df_tra, df_val, df_tes], ['tra', 'val', 'tes']):
            trans_multimodal(args_dict['dataset_path'], df, tvt_type, fold, args_dict, models_dict, device)
            
    print(f"\n[Done] Data generation completed. Control returned to the main program.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Offline feature extraction script')
    parser.add_argument('--dataset_path', type=str, default='data/RSID')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_seq_len', type=int, default=267) ##220
    parser.add_argument('--max_mol_len', type=int, default=241)
    
    parser.add_argument('--use_rna_2d', type=int, default=1)
    parser.add_argument('--use_mol_sem', type=int, default=1)
    parser.add_argument('--use_rna_sem', type=int, default=1)
    parser.add_argument('--split_mode', type=str, default='random',
                        choices=['random', 'cold_rna', 'cold_drug', 'cold_both'])
    
    parser.add_argument('--mol_chem_path', type=str, default='./pretrained_models/ChemBERTa-77M-MTR')
    parser.add_argument('--mol_sem_path', type=str, default='./pretrained_models/biobert-base-cased-v1.2')
    parser.add_argument('--rnabert_path', type=str, default='./pretrained_models/rnabert')
    
    args = parser.parse_args()
    
    run_generation(args)