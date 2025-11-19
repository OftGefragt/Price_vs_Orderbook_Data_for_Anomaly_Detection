
window_len = 30            # sliding window length
step = 1                   # step size for sliding windows
batch_size = 64            # dataloader batch size
scaling_range = (-1, 1)    # MinMaxScaling range
expected_features = 14     # multivariate time-series dimensionality


#GAN configs
z_dim = 100            # latent space dimension
gf_dim = 128           # generator base filters
df_dim = 32            # discriminator base filters
channels = 14          # GAN input channels = number of time-series features


#Training settings
epochs_gan = 10        # GAN training epochs
epochs_enc = 20        # Encoder training epochs
lr_gan = 2e-4          # Adam lr for generator
lr_disc = 1e-4         # Adam lr for discriminator
lr_enc = 1e-4          # Adam lr for encoder
betas = (0.5, 0.999)   # Adam betas for stability


alpha_rec = 1.0        # reconstruction loss weight
beta_feat = 0.1        # feature-matching loss weight


#anomaly scoring
anomaly_alpha = 0.9   

#optional noise injection -- not implemented in the ready model -- 
'''
def add_noise(x, std=0.05):
    """Apply Gaussian noise to stabilise D training."""
    return x + torch.randn_like(x) * std
'''


bce_logits = nn.BCEWithLogitsLoss()  # loss for GAN
mse_loss = nn.MSELoss()              # for encoder reconstruction + feature matching

