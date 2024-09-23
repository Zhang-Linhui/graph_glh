import torch
import torch.nn.functional as F

class LSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers, out_channels):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(in_channels, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, out_channels)

    def forward(self, data):
        x = data.x.unsqueeze(1)  # 添加一个维度，使其成为 (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return F.log_softmax(out, dim=1)
