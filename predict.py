import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, AutoModel
from model import MultiModalModel
from create_data import SimpleRnaTokenizer, extract_feature
from utils import rna2D_from_dot

# --- 1. Configuration and parameter settings ---
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
    'model_weight': 'model/RSID_random_fold1.pt' # Ensure this path points to the best-trained model
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_feature_extractors():
    print("[Init] Loading feature extraction models (this may take some time)...")
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
    """Convert a single row of Excel data to the Data object required by the model."""
    t_chem, t_bio, t_rna = tokenizers
    m_chem, m_bio, m_rna = models
    
    smiles = str(row['SMILES']) if pd.notna(row['SMILES']) else ""
    mol_text = str(row['Small molecule information']) if pd.notna(row['Small molecule information']) else ""
    rna_seq = str(row['1D Sequence']) if pd.notna(row['1D Sequence']) else ""
    rna_text = str(row['RNA information']) if pd.notna(row['RNA information']) else ""
    dot_bracket = str(row['Dot bracket']) if pd.notna(row['Dot bracket']) else ""

    with torch.no_grad():
        # Small molecule features
        mc_in = t_chem(smiles, max_length=args['max_mol_len'], padding='max_length', truncation=True, return_tensors='pt')
        mol_chem_feat = extract_feature(m_chem, mc_in, device)
        
        ms_in = t_bio(f"{smiles} [SEP] {mol_text}", max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        mol_sem_feat = extract_feature(m_bio, ms_in, device)
        
        # RNA features
        rf_in = t_rna(rna_seq, max_length=args['max_seq_len'], padding='max_length', truncation=True, return_tensors='pt')
        rna_bert_feat = extract_feature(m_rna, rf_in, device)
        
        rs_in = t_bio(f"{rna_seq} [SEP] {rna_text}", max_length=args['max_seq_len'], padding='max_length', truncation=True, return_tensors='pt')
        rna_sem_feat = extract_feature(m_bio, rs_in, device)

        # 2D structure
        res_2d = rna2D_from_dot(rna_seq[:args['max_seq_len']].ljust(args['max_seq_len'], 'N'), 
                                 dot_bracket[:args['max_seq_len']].ljust(args['max_seq_len'], '.'))
        rna_2d_x, rna_2d_edge_index, rna_2d_edge_type = res_2d

    # Build Mask (same logic as in training.py)
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
        y=torch.tensor([0.0]) # Placeholder
    )

def main():
    # 1. Prepare paths
    input_path = 'prediction/example.xlsx'
    output_path = 'prediction/prediction_results.xlsx'
    if not os.path.exists('prediction'): os.makedirs('prediction')

    # 2. Load data
    df = pd.read_excel(input_path)
    print(f"[Data] Loaded prediction task, {len(df)} rows of data.")

    # 3. Initialize feature extractors and model
    tokenizers, extractors = load_feature_extractors()
    
    # Dynamically set input dimensions
    args['mol_input_dim'] = 384 # ChemBERTa default
    args['rna_input_dim'] = 120 # RNABERT default
    args['bio_input_dim'] = 768 # BioBERT default
    
    model = MultiModalModel(args).to(device)
    model.load_state_dict(torch.load(args['model_weight'], map_location=device))
    model.eval()

    # 4. Perform prediction
    results = []
    print("[Predict] Performing inference...")
    for idx, row in df.iterrows():
        data_obj = process_row(row, tokenizers, extractors).to(device)
        # Add batch attribute (required by RGCN)
        data_obj.batch = torch.zeros(data_obj.rna_2d_x.size(0), dtype=torch.long).to(device)
        
        with torch.no_grad():
            output = model(data_obj)
            prob = torch.sigmoid(output['out']).item()
            results.append(prob)
        
        if (idx + 1) % 10 == 0: print(f"  Completed: {idx + 1}/{len(df)}")

    # 5. Save results
    df['Interaction_Probability'] = results  
    df['Prediction'] = (df['Interaction_Probability'] >= 0.5).astype(int)  # Generate predictions based on threshold
    df.drop(columns=['Interaction_Probability'], inplace=True)  
    df.to_excel(output_path, index=False)
    print(f"\n[Done] Prediction results saved to: {output_path}")

if __name__ == "__main__":
    main()