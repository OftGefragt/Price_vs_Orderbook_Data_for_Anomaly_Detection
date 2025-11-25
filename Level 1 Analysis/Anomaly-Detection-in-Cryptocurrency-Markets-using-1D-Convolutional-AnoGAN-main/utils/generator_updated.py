import torch.nn as nn
import config as config

class Generator1D(nn.Module):
    def __init__(self, z_dim=100, gf_dim=128, seq_len=config.SEQ_LEN, channels=config.CHANNELS):
        super().__init__()
        # design to upsample from spatial 1 -> seq_len via ConvTranspose stack
        self.net = nn.Sequential(
            nn.ConvTranspose1d(z_dim, gf_dim*4, 4, 1, 0),  # -> length 4
            nn.BatchNorm1d(gf_dim*4), nn.ELU(True),
            nn.ConvTranspose1d(gf_dim*4, gf_dim*2, 4, 2, 1),  # x2
            nn.BatchNorm1d(gf_dim*2), nn.ELU(True),
            nn.ConvTranspose1d(gf_dim*2, gf_dim, 4, 2, 1),    # x2
            nn.BatchNorm1d(gf_dim), nn.ELU(True),
            nn.ConvTranspose1d(gf_dim, channels, 4, 2, 1),    # x2
            nn.Tanh()
        )
    def forward(self, z): return self.net(z)
