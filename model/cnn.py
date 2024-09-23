'''
Author: Zhang-Linhui
Date: 2024-09-23 19:48:11
LastEditors: Zhang-Linhui
LastEditTime: 2024-09-23 20:05:07
'''
import torch
import torch.nn.functional as F

class CNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(64 * 4, 128)
        self.fc2 = torch.nn.Linear(128, out_channels)

    def forward(self, data):
        x = data.x.view(-1, 1, data.num_features)  # 假设输入是一维特征
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
