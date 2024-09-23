'''
Author: Zhang-Linhui
Date: 2024-09-23 07:07:52
LastEditors: Zhang-Linhui
LastEditTime: 2024-09-23 07:11:30
'''
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_data(dir="data"):
    # 加载数据
    x=np.load(f'{dir}/features.npy')
    y=np.load(f'{dir}/labels.npy')
    adj_matrix=np.load(f'{dir}/adjacency_matrix.npy')

    # 构建 PyG 的 Data 对象
    edge_index = torch.LongTensor(adj_matrix.T)
    x = torch.FloatTensor(x)
    y = torch.LongTensor(y)
    num_classes = len(torch.unique(y))
    
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_classes = num_classes

    # 创建训练和测试掩码
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)  # type: ignore
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)   # type: ignore

    # 示例：前半部分节点作为训练集，后半部分作为测试集
    split_idx = int(data.num_nodes * 0.8) # type: ignore
    data.train_mask[:split_idx] = True  # type: ignore
    data.test_mask[split_idx:] = True   # type: ignore


    return data
