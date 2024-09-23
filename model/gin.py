'''
Author: Zhang-Linhui
Date: 2024-09-23 07:36:48
LastEditors: Zhang-Linhui
LastEditTime: 2024-09-23 07:36:52
'''
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv

class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GIN, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(in_channels, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16))
        nn2 = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, out_channels))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
