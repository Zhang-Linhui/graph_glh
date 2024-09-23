
import torch
import torch.nn.functional as F
from torch_geometric.nn import SGConv

class SGC(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SGC, self).__init__()
        self.conv1 = SGConv(in_channels, out_channels, K=2)  # SGC 不需要激活函数

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)
