import torch
from utils.gru import GRUEncoder
from utils.inference import execute_inference, compute_anomaly_scores
from utils.data_preparation import create_test_sequence
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_test, test_data = create_test_sequence(config.TEST_PATH, time_step=10, load_scaler=False)

input_dim = X_test.shape[2]  # number of features per timestep

gru_encoder = GRUEncoder(input_dim=input_dim,embedding_dim=config.EMBEDDING_DIM,n_layers=config.N_LAYERS, dropout=config.DROPOUT_RATE)
gru_encoder.load_state_dict(torch.load(config.GRU_ENCODER_PATH))

c = torch.load(config.SVDD_PATH)

#getting the scores
execute_inference(gru_encoder, c, X_test, test_data, config.TIME_STEP, config.TOP_K, config.PERCENTILE, visualize_cluster=True)

import seaborn as sns
from tqdm import tqdm

# --- 1. Setup & Baseline ---
print("Calculating Baseline SVDD Scores...")
# We use your existing function to get the base scores
baseline_scores, _ = compute_anomaly_scores(gru_encoder, X_test, c, device='cpu')
baseline_scores = baseline_scores.cpu().numpy()

# --- 2. Feature Mapping ---
# Ensure feature names match the tensor dimensions
feature_names = test_data.columns.tolist()

# Handle mismatch if test_data has extra cols (like index/timestamps not in tensor)
if len(feature_names) != X_test.shape[2]:
    print(f"Warning: Name mismatch. DF has {len(feature_names)}, Tensor has {X_test.shape[2]}. Using indices.")
    feature_names = [f"Feature_{i}" for i in range(X_test.shape[2])]

# --- 3. Calculate Importance ---
importance_distributions = {}

print("Calculating impact distributions...")

# Iterate through every feature dimension
for i, col_name in enumerate(tqdm(feature_names)):
    # A. Clone the original tensor to avoid modifying it
    X_permuted = X_test.clone()

    # B. Shuffle ONLY this feature across the Batch dimension
    # This takes the trajectory of Feature 'i' from random other windows
    # and pastes it into the current window.
    # This breaks the temporal and cross-feature correlations.
    idx = torch.randperm(X_permuted.shape[0])
    X_permuted[:, :, i] = X_permuted[idx, :, i]

    # C. Run Inference
    perm_scores, _ = compute_anomaly_scores(gru_encoder, X_permuted, c, device='cpu')
    perm_scores = perm_scores.cpu().numpy()

    # D. Calculate Impact (Absolute change in distance to center)
    score_diffs = np.abs(baseline_scores - perm_scores)

    importance_distributions[col_name] = score_diffs

# --- 4. Process Results ---
df_imp = pd.DataFrame(importance_distributions)

# Sort by Median importance
sorted_cols = df_imp.median().sort_values(ascending=False).index
df_imp = df_imp[sorted_cols]

# --- 5. Visualization ---

# Plot 1: Linear Scale (General Importance)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_imp, orient='h', palette="vlag", fliersize=2)
plt.title('GRU with Deep SVDD Feature Importance\n(Impact on Distance to Center)')
plt.xlabel('Change in Anomaly Score (Euclidean Distance^2)')
plt.ylabel('Features')
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 2: Log Scale (To see outliers/anomalies better)
plt.figure(figsize=(12, 6))
g = sns.boxplot(data=df_imp, orient='h', palette="vlag", fliersize=2)
g.set_xscale("log")
plt.title('GRU with Deep SVDD Feature Importance (Log Scale)')
plt.xlabel('Change in Anomaly Score (Log Scale)')
plt.ylabel('Features')
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.show()