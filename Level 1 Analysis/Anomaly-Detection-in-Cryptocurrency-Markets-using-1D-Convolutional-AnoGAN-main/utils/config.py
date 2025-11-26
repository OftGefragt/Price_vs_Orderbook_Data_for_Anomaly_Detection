import torch.nn as nn
from torch.ao.nn.quantized.functional import threshold

WINDOW_LEN = 30            # sliding window length
STEP = 1                   # step size for sliding windows
BATCH_SIZE = 64            # dataloader batch size
SCALING_RANGE = (-1, 1)    # MinMaxScaling range
EXPECTED_FEATURES = 5     # multivariate time-series dimensionality


TRAIN_DATA_PATH = "../../data/train/BTC_1min.csv"
TEST_DATA_PATH = "../../data/test/ETH_1min.csv"
DEVICE = "cpu"
SEQ_LEN = 30

#GAN configs
Z_DIM = 100            # latent space dimension
GF_DIM = 128           # generator base filters
DF_DIM = 32            # discriminator base filters
CHANNELS = 5          # GAN input channels = number of time-series features


#Training settings
EPOCHS_GAN = 10       # GAN training epochs
EPOCHS_EN = 20        # Encoder training epochs
LR_GAN = 2e-4          # Adam lr for generator
LR_DISC = 1e-4         # Adam lr for discriminator
LR_ENG = 1e-4          # Adam lr for encoder
BETAS = (0.5, 0.999)   # Adam betas for stability


ALPHA_REC = 1.0        # reconstruction loss weight
BETA_FEAT = 0.1        # feature-matching loss weight


#anomaly scoring
ANOMALY_ALPHA = 0.9

#optional noise injection -- not implemented in the ready model -- 
'''
def add_noise(x, std=0.05):
    """Apply Gaussian noise to stabilise D training."""
    return x + torch.randn_like(x) * std
'''


BCE_LOGITS = nn.BCEWithLogitsLoss()  # loss for GAN
MSE_LOSS = nn.MSELoss()              # for encoder reconstruction + feature matching

