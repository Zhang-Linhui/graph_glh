'''
Author: Zhang-Linhui
Date: 2024-09-23 19:49:16
LastEditors: Zhang-Linhui
LastEditTime: 2024-09-23 20:11:32
'''
import torch
import torch.nn.functional as F
from torch.nn import Transformer

class TransformerModel(torch.nn.Module):
    def __init__(self, in_channels, num_heads, num_layers, out_channels):
        super(TransformerModel, self).__init__()
        self.transformer = Transformer(d_model=in_channels, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc = torch.nn.Linear(in_channels, out_channels)

    def forward(self, data):
        src = data.x.unsqueeze(0)  # 添加一个维度，使其成为 (sequence_length, batch_size, input_size)
        tgt = src  # 对于自回归模型，目标序列可以是源序列的一个偏移版本
        out = self.transformer(src, tgt)
        out = self.fc(out[-1, :, :])
        return F.log_softmax(out, dim=1)
