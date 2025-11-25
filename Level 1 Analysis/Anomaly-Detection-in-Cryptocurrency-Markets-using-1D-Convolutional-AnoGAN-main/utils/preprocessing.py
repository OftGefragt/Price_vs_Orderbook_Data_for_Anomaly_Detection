import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import config as config
import pandas as pd

def sliding_windows(arr, window=30, step=1):
    T, F = arr.shape
    if T < window:
        raise ValueError(f"time length {T} < window {window}")
    n = 1 + (T - window) // step
    out = np.stack([arr[i*step : i*step + window] for i in range(n)], axis=0)  # [N, window, F]
    return out  # [N, window, F]

def prepare_df(path):
    df = pd.read_csv(path)
    df['Timestamp'] = pd.to_datetime(df['Unix Timestamp'], unit='s')
    df.set_index('Timestamp', inplace=True)

    df.dropna(inplace=True)
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return df[feature_columns].copy()

df = prepare_df(config.TRAIN_DATA_PATH)
arr = df.values  # [T, F]
F = arr.shape[1]

windows = sliding_windows(arr, window=config.WINDOW_LEN, step=config.STEP)  # [N, window, channels]
# convert to [N, channels, seq_len]
X = windows.transpose(0,2,1).astype(np.float32)  # [N, channels, window_len]
print("windows shape", windows.shape, "X shape", X.shape)

scaler = MinMaxScaler(feature_range=(-1,1))
# flatten to [N*window_len, features]
flat = windows.reshape(-1, F)
flat_scaled = scaler.fit_transform(flat)
windows_scaled = flat_scaled.reshape(windows.shape)  # [N, window, F]
X_scaled = windows_scaled.transpose(0,2,1)  # [N, 14, window_len]

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

#channels = 14 #chosen most important features -- adjust accordingly
seq_len = config.WINDOW_LEN
#print("final tensor shape", X_tensor.shape, "channels", channels, "seq_len", seq_len)

# save scaler
with open('../models/scaler.save', 'wb') as f:
    pickle.dump(scaler, f)