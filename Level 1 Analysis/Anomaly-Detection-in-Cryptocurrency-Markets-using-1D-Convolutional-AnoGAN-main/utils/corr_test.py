import numpy as np
#import pandas as pd, numpy as np, json
#the btc dataset in the train folder has more features than the amount in the trained model.
#we use a shap and correlation test to get the top features.
#over %80 means the feature is dropped from the dataset.

def select_features_by_corr(df, threshold=0.80, verbose=True):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    reduced = df.drop(columns=to_drop)
    kept = list(reduced.columns)
    if verbose:
        print(f"Kept {len(kept)} features, dropped {len(to_drop)}")
    return reduced, kept, to_drop

