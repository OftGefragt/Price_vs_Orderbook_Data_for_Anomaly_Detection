import pandas as pd
# extract important features from data
def prepare_df(path):
    df = pd.read_csv(path)
    df['Timestamp'] = pd.to_datetime(df['Unix Timestamp'], unit='s')
    df.set_index('Timestamp', inplace=True)

    df.dropna(inplace=True)
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return df[feature_columns].copy()
