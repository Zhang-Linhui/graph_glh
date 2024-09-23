import torch
import torch.nn.functional as F
from torch_geometric.nn import APPNP

class APPNPNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K=10, alpha=0.1):
        super(APPNPNet, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, 16)
        self.lin2 = torch.nn.Linear(16, out_channels)
        self.prop = APPNP(K, alpha)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)
