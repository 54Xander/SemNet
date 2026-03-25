import os
import torch
import numpy as np
import random
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score, average_precision_score
import torch
from torch_geometric.data import InMemoryDataset
import torch.nn.functional as F

def set_seed(seed):
    """
    全方位设置随机种子以确保结果可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 环境变量设置
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 针对 CUDA 10.2+ 的矩阵乘法确定性设置
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # 强制 CuDNN 使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 强制 PyTorch 使用确定性算法
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass

def rna2D_from_dot(seq, dot_bracket):
    base_dict = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
    indices = torch.tensor([base_dict.get(b.upper(), 4) for b in seq], dtype=torch.long)
    x = F.one_hot(indices, num_classes=5).float() 

    stack, pair_map = [], {}
    for i, c in enumerate(dot_bracket):
        if c == '(': stack.append(i)
        elif c == ')':
            if stack:
                j = stack.pop()
                pair_map[i], pair_map[j] = j, i

    edge_type_map = {'link': 0, ('C', 'G'): 1, ('A', 'U'): 2, ('G', 'U'): 3, 
                    ('A', 'G'): 4, ('U', 'U'): 5, ('C', 'C'): 6, ('A', 'A'): 7, 'unknown': 8}
    edge_index, edge_type = [], []

    for i in range(len(seq) - 1):
        edge_index.extend([[i, i + 1], [i + 1, i]])
        edge_type.extend([0, 0])

    for i, j in pair_map.items():
        if i >= j: continue
        sorted_bases = tuple(sorted([seq[i].upper(), seq[j].upper()]))
        pair_type = edge_type_map.get(sorted_bases, 8)
        edge_index.extend([[i, j], [j, i]])
        edge_type.extend([pair_type, pair_type])

    if not edge_index:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        
    return x, edge_index, edge_type

def get_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        auc_score = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc_score = 0.5 

    aupr_score = average_precision_score(y_true, y_pred)
    pred_label = (y_pred >= 0.5).astype(int)
    tn = ((pred_label == 0) & (y_true == 0)).sum()
    fp = ((pred_label == 1) & (y_true == 0)).sum()
    spec = tn / (tn + fp + 1e-7)
    
    # 【修改点】在这里加上 float() 强制转换，解决 np.float64 显示问题
    return [
        round(float(auc_score), 4), 
        round(float(aupr_score), 4), 
        round(float(f1_score(y_true, pred_label)), 4), 
        round(float(accuracy_score(y_true, pred_label)), 4), 
        round(float(recall_score(y_true, pred_label)), 4), 
        round(float(spec), 4), 
        round(float(precision_score(y_true, pred_label, zero_division=0)), 4)
    ]

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', xd=None, xt=None, y=None, 
                 transform=None, pre_transform=None, smile_graph=None, k_mer_features=None):
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
