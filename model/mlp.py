'''
Author: Zhang-Linhui
Date: 2024-09-23 19:47:52
LastEditors: Zhang-Linhui
LastEditTime: 2024-09-23 19:47:59
'''
import torch
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, out_channels)

    def forward(self, data):
        x = data.x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
