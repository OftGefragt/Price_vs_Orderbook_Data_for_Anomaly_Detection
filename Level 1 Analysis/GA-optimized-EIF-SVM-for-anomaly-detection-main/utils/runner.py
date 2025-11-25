import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.eif_data_preparation import prepare_df
from utils.eif_visualization import plot_eif_anomaly_scores, plot_anomaly_context
import config

X_test_df = prepare_df(config.TEST_PATH)

with open(config.EIF_PATH, "rb") as f:
    forest = pickle.load(f)

with open(config.OC_SVM_PATH, "rb") as f:
    oc_svm = pickle.load(f)

scaler = MinMaxScaler(feature_range=(0, 1))
X_test_scaled_numpy = scaler.fit_transform(X_test_df)

eif_scores = forest.compute_paths(X_in=X_test_scaled_numpy)

scores_reshaped = eif_scores.reshape(-1, 1)

svm_preds = oc_svm.predict(scores_reshaped)


results_df = X_test_df.copy()

results_df['eif_score'] = eif_scores

anomaly_threshold = results_df['eif_score'].quantile(config.THRESHOLD)
results_df['is_anomaly'] = results_df['eif_score'] >= anomaly_threshold

# 1 normal, -1 anomaly
results_df['prediction'] = np.where(results_df['is_anomaly'], -1, 1)

n_anomalies = np.sum(results_df['is_anomaly'])

print("Analysis complete. Number of anomalies found:", n_anomalies)

top_anomalies = results_df[results_df['is_anomaly']].nlargest(config.TOP_K, 'eif_score')

for i, (timestamp, row) in enumerate(top_anomalies.iterrows()):
    score = row['eif_score']
    print(f"Timestamp: {timestamp}, Anomaly Score: {score:.4f}")
    # Pass the original unscaled DataFrame for plotting
    plot_anomaly_context(X_test_df, timestamp)

plot_eif_anomaly_scores(results_df, anomaly_threshold)