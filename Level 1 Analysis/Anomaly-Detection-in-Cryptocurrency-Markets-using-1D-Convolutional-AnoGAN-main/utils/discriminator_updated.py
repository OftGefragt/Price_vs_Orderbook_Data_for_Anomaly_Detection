import torch.nn as nn
import config as config

class Discriminator1D(nn.Module):
    def __init__(self, c_dim=config.CHANNELS, df_dim=32):
        super().__init__()
        self.conv0 = nn.Conv1d(c_dim, df_dim, 4, 2, 1, bias=False)
        self.elu0 = nn.ELU(True)
        self.conv1 = nn.Conv1d(df_dim, df_dim*2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(df_dim*2)
        self.elu1 = nn.ELU(True)
        # final conv reduces to small spatial size -> map to single-channel map
        self.conv2 = nn.Conv1d(df_dim*2, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        h0 = self.elu0(self.conv0(x))
        h1 = self.elu1(self.bn1(self.conv1(h0)))
        h2 = self.conv2(h1)
        feat = h1.mean(dim=2)      # [B, C]
        out = h2.mean(dim=2).view(-1)  # scalar per sample
        return feat, out
