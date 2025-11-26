import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch.nn as nn
from utils.generator_updated import Generator1D
from utils.discriminator_updated import Discriminator1D
from utils.encoder import Encoder1D
from utils.preprocessing import prepare_df, sliding_windows
from utils.visualization import plot_anomaly_context, plot_anogan_anomaly_scores
import config as config

# Create models with the same architecture as during training
G = Generator1D(z_dim=config.Z_DIM).to(config.DEVICE)
D = Discriminator1D().to(config.DEVICE)
E = Encoder1D(z_dim=config.Z_DIM, c_dim=config.CHANNELS, seq_len=config.WINDOW_LEN).to(config.DEVICE)

# Load model weights
G.load_state_dict(torch.load("../models/generator.pth", map_location=config.DEVICE))
D.load_state_dict(torch.load("../models/discriminator.pth", map_location=config.DEVICE))
E.load_state_dict(torch.load("../models/encoder.pth", map_location=config.DEVICE))

# Set models to evaluation mode
G.eval()
D.eval()
E.eval()

# Create test data loader
print("Preparing test data...")
test_df = prepare_df(config.TEST_DATA_PATH)
test_arr = test_df.values # get numpy array from dataframe
scaler = MinMaxScaler(feature_range=(0, 1))
test_arr_scaled = scaler.fit_transform(test_arr) # scale data to test data, the ETH is different from BTC data
test_windows = sliding_windows(test_arr_scaled, window=config.WINDOW_LEN, step=config.STEP) # create sliding windows
X_test_tensor = torch.tensor(test_windows.transpose(0, 2, 1), dtype=torch.float32) # convert to tensor and change shape to [N, channels, seq_len]
test_dataset = torch.utils.data.TensorDataset(X_test_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False) # as it is time series, we don't want to shuffle the data

# initialization of variables
anomaly_scores = []
mse = nn.MSELoss(reduction='none')

alpha = 1.0
beta = 0.1

with torch.no_grad():
    for (batch,) in tqdm(test_loader):
        batch = batch.to(config.DEVICE)
        # Generate reconstruction and features
        z_enc = E(batch) # encode to latent space
        rec = G(z_enc) # reconstruct from latent space
        # in case of size mismatch due to conv layers
        if rec.size(2) != batch.size(2):
            rec = rec[:, :, :batch.size(2)] # trim reconstruction to match input size
        # Compute reconstruction loss
        loss_rec = mse(rec, batch).mean(dim=[1, 2])

        f_real, _ = D(batch) # discriminate features from real data
        f_rec, _ = D(rec) # discriminate features from reconstructed data
        loss_feat = mse(f_rec, f_real).mean(dim=1) # calculate feature-matching loss

        score = alpha * loss_rec + beta * loss_feat # weighting losses to get final anomaly score

        anomaly_scores.extend(score.cpu().numpy()) # store scores

anomaly_scores = np.array(anomaly_scores) # convert tensor to numpy array

results_df = pd.DataFrame(
    index=test_df.index[config.WINDOW_LEN - 1:], # adjust index to match sliding windows
    data={'anomaly_score': anomaly_scores}
)

anomaly_threshold = results_df['anomaly_score'].quantile(0.95) # set the threshold at 95th percentile
results_df['is_anomaly'] = results_df['anomaly_score'] >= anomaly_threshold # mark anomalies

n_anomalies = results_df['is_anomaly'].sum() # calculate number of anomalies
print(f"Number of anomalies found: {n_anomalies}")


top_anomalies = results_df[results_df['is_anomaly']].nlargest(3, 'anomaly_score') # get top 3 anomalies
# display top 3 anomalies
for i, (timestamp, row) in enumerate(top_anomalies.iterrows()):
    score = row['anomaly_score']
    print(f"Timestamp: {timestamp}, Anomaly Score: {score:.4f}")
    plot_anomaly_context(test_df, timestamp)
# plot anomaly score for whole timeframe of test set
plot_anogan_anomaly_scores(results_df, anomaly_threshold)