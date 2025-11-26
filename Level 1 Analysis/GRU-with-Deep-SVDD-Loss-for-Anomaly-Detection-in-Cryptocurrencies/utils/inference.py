import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def compute_anomaly_scores(gru_encoder, X_new, c, device='cpu'):
    gru_encoder.to(device)
    X_new = X_new.to(device)
    c = c.to(device)

    with torch.no_grad():
        embeddings = gru_encoder(X_new) # get embeddings
        diff = embeddings - c # compute difference from center
        scores = torch.sum(diff**2, dim=1) # compute squared distance as anomaly score

    return scores, embeddings

def detect_anomalies(anomaly_scores, percentile=95):
    threshold = torch.quantile(anomaly_scores, percentile / 100.0) # compute threshold
    anomalies = (anomaly_scores > threshold) # detect anomalies
    return anomalies, threshold

def plot_anomaly_scores(anomaly_scores, anomalies, threshold):
    scores_np = anomaly_scores.cpu().numpy()
    anomalies_np = anomalies.cpu().numpy()
    threshold_np = threshold.cpu().numpy()

    plt.figure(figsize=(12, 5))
    
    normal_indices = ~anomalies_np
    anomaly_indices = anomalies_np

    plt.scatter(np.where(normal_indices)[0], scores_np[normal_indices], 
                c='blue', label='Normal', alpha=0.6)
    plt.scatter(np.where(anomaly_indices)[0], scores_np[anomaly_indices], 
                c='red', label='Anomaly', alpha=0.8)
    
    plt.axhline(threshold_np, color='black', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.xlabel("Sequence Index")
    plt.ylabel("Anomaly Score")
    plt.title("Deep SVDD Anomaly Detection Results")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def plot_anomaly_context(test_data, anomaly_timestamp, window_minutes=30):
    half_window = pd.Timedelta(minutes=window_minutes / 2) # calculate half window i.e. 15 min
    start_time = anomaly_timestamp - half_window
    end_time = anomaly_timestamp + half_window

    # Select the data for this specific window
    context_df = test_data.loc[start_time:end_time]

    if context_df.empty:
        print(f"No data found for the window around {anomaly_timestamp}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    # Plot price data
    ax1.plot(context_df.index, context_df['Close'], label='Close Price', color='blue')
    ax1.fill_between(context_df.index, context_df['Low'], context_df['High'],
                     color='gray', alpha=0.3, label='High-Low Range')
    ax1.axvline(anomaly_timestamp, color='red', linestyle='--', lw=2, label='Anomaly Detected')
    ax1.set_title(f"Market Context Around Anomaly: {anomaly_timestamp}")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    ax1.grid(True)
    # Plot volume data
    ax2.bar(context_df.index, context_df['Volume'], width=0.0005, color='green', label='Volume')
    ax2.axvline(anomaly_timestamp, color='red', linestyle='--', lw=2)
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Timestamp")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
# funciton to automate inference process
def execute_inference(gru_encoder, c, X_test, test_data, time_step=10, top_k=3, percentile=95):
    anomaly_scores, embeddings = compute_anomaly_scores(gru_encoder, X_test, c, device='cpu') # compute scores
    anomalies, threshold = detect_anomalies(anomaly_scores, percentile=95) # detect anomalies

    top_scores, top_indices = torch.topk(anomaly_scores, k=top_k) # get top k anomalies

    for i in range(top_k):
        idx = top_indices[i].item()
        score = top_scores[i].item()
        timestamp_idx = idx + time_step - 1

        if timestamp_idx < len(X_test):
            anomaly_timestamp = test_data.index[timestamp_idx]
            print(f"Anomaly {i + 1}: Timestamp: {anomaly_timestamp}, Anomaly Score: {score:.10f}")
            plot_anomaly_context(test_data, anomaly_timestamp, window_minutes=30)
        else:
            print(f"Anomaly {i + 1}: Anomaly Score: {score:.10f}, Index out of bounds for timestamp retrieval.")

    print(f"Plotting results with a threshold at the {percentile}th percentile...")
    plot_anomaly_scores(anomaly_scores, anomalies, threshold)

