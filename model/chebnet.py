import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class ChebNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K=3):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_channels, 16, K)
        self.conv2 = ChebConv(16, out_channels, K)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

