import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def prepare_df(path):
    df = pd.read_csv(path)
    df['Timestamp'] = pd.to_datetime(df['Unix Timestamp'], unit='s')
    df.set_index('Timestamp', inplace=True)

    df.dropna(inplace=True)
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[feature_columns].copy()
    return df

def split_test_train(df, split_ratio=0.8):
    num_train = int(len(df) * split_ratio)
    train= df[:num_train]
    test = df[num_train:]
    return train, test

def scale_data(train_data, test_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)

    # Save scaler for future new data
    joblib.dump(scaler, '../model/scaler.save')
    return scaled_train_data, scaled_test_data

def create_sequences(data, time_step=10):
    X = []
    for i in range(len(data) - time_step + 1):
        X.append(data[i:i+time_step])
    return np.array(X)

def create_train_test_sequences(path, split_ratio=0.8, time_step=10):
    df_features = prepare_df(path)
    train_data, test_data = split_test_train(df_features, split_ratio)
    scaled_train_data, scaled_test_data = scale_data(train_data, test_data)

    X_train = create_sequences(scaled_train_data, time_step=time_step)
    X_test = create_sequences(scaled_test_data, time_step=time_step)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    return X_train, X_test, test_data

def create_test_sequence(path, time_step=10, load_scaler=False):
    df_features = prepare_df(path)
    if load_scaler:
        scaler = joblib.load('../model/scaler.save')
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_features)

    X_test = create_sequences(scaled_data, time_step=time_step)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    return X_test, df_features

