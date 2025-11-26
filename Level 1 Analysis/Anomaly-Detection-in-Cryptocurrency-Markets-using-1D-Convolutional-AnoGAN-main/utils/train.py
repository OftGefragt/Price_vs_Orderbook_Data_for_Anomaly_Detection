import torch
from utils.discriminator_updated import Discriminator1D
from utils.encoder import Encoder1D
from utils.generator_updated import Generator1D
import torch.nn as nn
import torch.optim as optim
from preprocessing import loader, F
import config as config

# create models with defined architectures
G = Generator1D(z_dim=config.Z_DIM).to(config.DEVICE)
D = Discriminator1D().to(config.DEVICE)
E = Encoder1D(z_dim=config.Z_DIM, c_dim=F, seq_len=config.WINDOW_LEN).to(config.DEVICE)

#Checking the shapes
with torch.no_grad():
    z = torch.randn(4, config.Z_DIM, 1, device=config.DEVICE) # batch of 4 random latent vectors
    g_out = G(z) # generate data from latent vectors
    print("G out", g_out.shape)   # expect [4, channels, seq_len]
    x_sample = next(iter(loader))[0][:4].to(config.DEVICE)
    z_e = E(x_sample) # encode sample data
    f_real, o_real = D(x_sample) # discriminate sample data
    print("E out", z_e.shape, "D feat", f_real.shape, "D out", o_real.shape)
    assert g_out.shape[1] == config.CHANNELS, "generator channels mismatch"
    assert g_out.shape[2] >= config.SEQ_LEN-4 and g_out.shape[2] <= config.SEQ_LEN+4, "check generator output length (may need layer tweaks)"

#Training the GAN
bce = nn.BCEWithLogitsLoss()
G_opt = optim.Adam(G.parameters(), lr=4e-4, betas=(0.5,0.999))
D_opt = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5,0.999))

for epoch in range(config.EPOCHS_GAN):
    for (real,) in loader:
        real = real.to(config.DEVICE)
        bs = real.size(0) # batch size
        # Train D
        D_opt.zero_grad()
        _, real_logits = D(real) # discriminate real data
        real_labels = torch.ones(bs, device=config.DEVICE)
        loss_real = bce(real_logits, real_labels) # calculate bce loss for real data
        z = torch.randn(bs, config.Z_DIM, 1, device=config.DEVICE)
        fake = G(z) # generate fake data
        _, fake_logits = D(fake.detach())
        fake_labels = torch.zeros(bs, device=config.DEVICE)
        loss_fake = bce(fake_logits, fake_labels) # calculate bce loss for fake data
        lossD = 0.5*(loss_real + loss_fake) # weight the losses
        lossD.backward(); D_opt.step() # backpropagation and optimization
        # Train G
        G_opt.zero_grad()
        _, fake_logits2 = D(fake) # discriminate fake data
        lossG = bce(fake_logits2, real_labels) # caluclate loss against real labels
        lossG.backward(); G_opt.step() # backpropagation and optimization
    print(f"GAN Epoch {epoch+1}/{config.EPOCHS_GAN}  D {lossD.item():.4f}  G {lossG.item():.4f}")

#Training the encoder
E_opt = optim.Adam(E.parameters(), lr=1e-4, betas=(0.5,0.999))
mse = nn.MSELoss()
alpha = 1.0
beta = 0.1
# Train encoder
for epoch in range(config.EPOCHS_EN):
    for (real,) in loader:
        real = real.to(config.DEVICE)
        E_opt.zero_grad()
        # Encode real data
        z_enc = E(real)
        # Reconstruct via G
        rec = G(z_enc)
        # crop if size mismatch
        if rec.size(2) != real.size(2):
            rec = rec[:, :, :real.size(2)]
        # Feature matching
        f_real, _ = D(real)
        f_rec, _ = D(rec)
        loss_rec = mse(rec, real) # reconstruction loss
        loss_feat = mse(f_rec, f_real) # feature matching loss
        lossE = alpha * loss_rec + beta * loss_feat # weight losses
        lossE.backward()
        E_opt.step()
    print(f"Encoder Epoch {epoch+1}/{config.EPOCHS_EN}  Loss {lossE.item():.4f}")

#Save models
torch.save(G.state_dict(), '../models/generator.pth')
torch.save(D.state_dict(), '../models/discriminator.pth')
torch.save(E.state_dict(), '../models/encoder.pth')