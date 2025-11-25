import torch.nn as nn
import config as config
import torch

class Encoder1D(nn.Module):
    def __init__(self, z_dim=100, c_dim=config.CHANNELS, ef_dim=64, seq_len=config.SEQ_LEN):
        super().__init__()
        self.conv0 = nn.Conv1d(c_dim, ef_dim, 4, 2, 1)
        self.elu0 = nn.ELU(True)
        self.conv1 = nn.Conv1d(ef_dim, ef_dim*2, 4, 2, 1)
        self.bn1 = nn.BatchNorm1d(ef_dim*2)
        self.elu1 = nn.ELU(True)
       
        with torch.no_grad():
            dummy = torch.zeros(1, c_dim, seq_len)
            h = self.elu1(self.bn1(self.conv1(self.elu0(self.conv0(dummy)))))
            flat = h.numel() // h.shape[0]
        self.fc = nn.Linear(flat, z_dim)
    def forward(self, x):
        h = self.elu0(self.conv0(x))
        h = self.elu1(self.bn1(self.conv1(h)))
        h = h.flatten(1)
        z = self.fc(h)
        return z.unsqueeze(2)  # [B, z_dim, 1]
