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


G = Generator1D(z_dim=config.Z_DIM).to(config.DEVICE)
D = Discriminator1D().to(config.DEVICE)
E = Encoder1D(z_dim=config.Z_DIM, c_dim=config.CHANNELS, seq_len=config.WINDOW_LEN).to(config.DEVICE)

G.load_state_dict(torch.load("../models/generator.pth", map_location=config.DEVICE))
D.load_state_dict(torch.load("../models/discriminator.pth", map_location=config.DEVICE))
E.load_state_dict(torch.load("../models/encoder.pth", map_location=config.DEVICE))

G.eval()
D.eval()
E.eval()


print("Preparing test data...")
test_df = prepare_df(config.TEST_DATA_PATH)
test_arr = test_df.values
scaler = MinMaxScaler(feature_range=(0, 1))
test_arr_scaled = scaler.fit_transform(test_arr)
test_windows = sliding_windows(test_arr_scaled, window=config.WINDOW_LEN, step=config.STEP)
X_test_tensor = torch.tensor(test_windows.transpose(0, 2, 1), dtype=torch.float32)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)


anomaly_scores = []
mse = nn.MSELoss(reduction='none')

alpha = 1.0
beta = 0.1

with torch.no_grad():
    for (batch,) in tqdm(test_loader):
        batch = batch.to(config.DEVICE)

        z_enc = E(batch)
        rec = G(z_enc)

        if rec.size(2) != batch.size(2):
            rec = rec[:, :, :batch.size(2)]

        loss_rec = mse(rec, batch).mean(dim=[1, 2])

        f_real, _ = D(batch)
        f_rec, _ = D(rec)
        loss_feat = mse(f_rec, f_real).mean(dim=1)

        score = alpha * loss_rec + beta * loss_feat

        anomaly_scores.extend(score.cpu().numpy())

anomaly_scores = np.array(anomaly_scores)

results_df = pd.DataFrame(
    index=test_df.index[config.WINDOW_LEN - 1:],
    data={'anomaly_score': anomaly_scores}
)

anomaly_threshold = results_df['anomaly_score'].quantile(0.95)
results_df['is_anomaly'] = results_df['anomaly_score'] >= anomaly_threshold

n_anomalies = results_df['is_anomaly'].sum()
print(f"Number of anomalies found: {n_anomalies}")


top_anomalies = results_df[results_df['is_anomaly']].nlargest(3, 'anomaly_score')

for i, (timestamp, row) in enumerate(top_anomalies.iterrows()):
    score = row['anomaly_score']
    print(f"Timestamp: {timestamp}, Anomaly Score: {score:.4f}")
    plot_anomaly_context(test_df, timestamp)

plot_anogan_anomaly_scores(results_df, anomaly_threshold)