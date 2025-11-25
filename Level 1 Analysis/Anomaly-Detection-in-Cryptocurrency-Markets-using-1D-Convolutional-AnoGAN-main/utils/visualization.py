import matplotlib.pyplot as plt
import pandas as pd

def plot_anogan_anomaly_scores(results_df, threshold):
    plt.figure(figsize=(15, 6))

    normal_df = results_df[results_df['is_anomaly'] == False]
    anomaly_df = results_df[results_df['is_anomaly'] == True]

    plt.scatter(normal_df.index, normal_df['anomaly_score'],
                c='blue', label='Normal', alpha=0.5, s=10)

    plt.scatter(anomaly_df.index, anomaly_df['anomaly_score'],
                c='red', label='Anomaly', alpha=0.8, s=30)

    plt.axhline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.4f})')

    plt.xlabel("Timestamp")
    plt.ylabel("AnoGAN Anomaly Score")
    plt.title("AnoGAN Anomaly Detection Results")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_anomaly_context(test_data, anomaly_timestamp, window_minutes=30):
    half_window = pd.Timedelta(minutes=window_minutes / 2)
    start_time = anomaly_timestamp - half_window
    end_time = anomaly_timestamp + half_window

    context_df = test_data.loc[start_time:end_time]

    if context_df.empty:
        print(f"No data found for the window around {anomaly_timestamp}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(context_df.index, context_df['Close'], label='Close Price', color='blue')
    ax1.fill_between(context_df.index, context_df['Low'], context_df['High'],
                     color='gray', alpha=0.3, label='High-Low Range')
    ax1.axvline(anomaly_timestamp, color='red', linestyle='--', lw=2, label='Anomaly Detected')
    ax1.set_title(f"Market Context Around Anomaly: {anomaly_timestamp}")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    ax1.grid(True)

    ax2.bar(context_df.index, context_df['Volume'], width=0.0005, color='green', label='Volume')
    ax2.axvline(anomaly_timestamp, color='red', linestyle='--', lw=2)
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Timestamp")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()