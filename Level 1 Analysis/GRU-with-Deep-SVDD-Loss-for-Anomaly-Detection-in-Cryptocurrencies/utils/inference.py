import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def compute_anomaly_scores(gru_encoder, X_new, c, device='cpu'):
    gru_encoder.to(device)
    X_new = X_new.to(device)
    c = c.to(device)

    with torch.no_grad():
        embeddings = gru_encoder(X_new)
        diff = embeddings - c
        scores = torch.sum(diff**2, dim=1)

    return scores, embeddings

def detect_anomalies(anomaly_scores, percentile=95):
    threshold = torch.quantile(anomaly_scores, percentile / 100.0)
    anomalies = (anomaly_scores > threshold)
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
    half_window = pd.Timedelta(minutes=window_minutes / 2)
    start_time = anomaly_timestamp - half_window
    end_time = anomaly_timestamp + half_window

    # Select the data for this specific window
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


def visualize_anomaly_clusters(X_data, anomaly_scores, threshold, save_plots=False):
    print("\n--- Starting Manifold Visualization ---")

    # 1. Convert inputs to Numpy
    if isinstance(X_data, torch.Tensor):
        X_data = X_data.detach().cpu().numpy()
    if isinstance(anomaly_scores, torch.Tensor):
        anomaly_scores = anomaly_scores.detach().cpu().numpy()
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.item()

    # 2. Flatten if 3D (e.g., [Batch, Time, Feat] -> [Batch, Time*Feat])
    if X_data.ndim > 2:
        N, W, C = X_data.shape
        X_viz = X_data.reshape(N, -1)
    else:
        X_viz = X_data

    # 3. Compute PCA & t-SNE
    print("Computing PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_viz)

    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_viz)

    # 4. Define Hard-Cut Colormap
    # Blue Gradient -> [Threshold] -> Red Gradient
    colors = [
        (0.0, "#00008B"),  # Deep Blue (Lowest Score)
        (0.5, "#4da6ff"),  # Light Blue (Approaching Threshold)
        (0.5, "#ff6666"),  # Light Red (Crossed Threshold)
        (1.0, "#8B0000")  # Deep Red (Highest Score)
    ]
    cmap_hard_cut = mcolors.LinearSegmentedColormap.from_list("BlueRedHardCut", colors)

    # --- HELPER: Dynamic Number Formatting ---
    def format_score(val):
        """Returns scientific notation if value is very small, else float."""
        if val == 0: return "0.00"
        if abs(val) < 0.01 or abs(val) > 1000:
            return f"{val:.2e}"  # Scientific: 1.25e-4
        return f"{val:.4f}"  # Standard: 0.1234

    # 5. Plotting Logic
    def _plot(X_2d, method_name, filename):
        plt.figure(figsize=(12, 9))

        # A. Sort indices so Red points (Anomalies) plot ON TOP of Blue points
        sorted_indices = np.argsort(anomaly_scores)
        X_sorted = X_2d[sorted_indices]
        scores_sorted = anomaly_scores[sorted_indices]

        # B. Normalization: Pin the center (0.5) to the Threshold
        div_norm = mcolors.TwoSlopeNorm(
            vmin=anomaly_scores.min(),
            vcenter=threshold,
            vmax=anomaly_scores.max()
        )

        # C. Scatter Plot
        scatter = plt.scatter(
            X_sorted[:, 0], X_sorted[:, 1],
            c=scores_sorted,
            cmap=cmap_hard_cut,
            norm=div_norm,
            alpha=0.7,
            s=25,
            edgecolors='none'
        )

        # D. Colorbar with Smart Ticks
        # Calculate 5 points: Min, Mid-Normal, Threshold, Mid-Anomaly, Max
        tick_locs = [
            anomaly_scores.min(),
            (anomaly_scores.min() + threshold) / 2,
            threshold,
            (threshold + anomaly_scores.max()) / 2,
            anomaly_scores.max()
        ]

        cbar = plt.colorbar(scatter, ticks=tick_locs)
        cbar.set_label('SVDD Anomaly Score (Intensity)', fontsize=12)

        # Apply the formatter to the ticks
        cbar.ax.set_yticklabels([format_score(t) for t in tick_locs])

        # E. Discrete Legend (Just for Class ID)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Normal Region',
                   markerfacecolor='#4da6ff', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Anomaly Region',
                   markerfacecolor='#ff6666', markersize=10),
        ]
        plt.legend(handles=legend_elements, loc='upper right', title="Classification")

        # F. Info Box
        textstr = f'Threshold: {format_score(threshold)}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
                       verticalalignment='top', bbox=props)

        plt.title(f'{method_name}: Deep SVDD Latent Space', fontsize=16)
        plt.grid(True, alpha=0.2)

        if save_plots:
            plt.savefig(filename)
            print(f"Saved {filename}")
        plt.show()

    _plot(X_pca, "PCA", "svdd_pca.png")
    _plot(X_tsne, "t-SNE", "svdd_tsne.png")


def execute_inference(gru_encoder, c, X_test, test_data, time_step=10, top_k=3, percentile=95, visualize_cluster=False):
    anomaly_scores, embeddings = compute_anomaly_scores(gru_encoder, X_test, c, device='cpu')
    anomalies, threshold = detect_anomalies(anomaly_scores, percentile=95)

    top_scores, top_indices = torch.topk(anomaly_scores, k=top_k)

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

    if(visualize_cluster):
        visualize_anomaly_clusters(
            X_data=embeddings,
            anomaly_scores=anomaly_scores,
            threshold=threshold,
            save_plots=True
        )
