from sklearn.preprocessing import MinMaxScaler
from eif.eif_class import iForest
import pickle
import numpy as np
from sklearn.svm import OneClassSVM
from utils.eif_data_preparation import prepare_df
import config

df = prepare_df(config.TRAIN_PATH)
df.head(2)
print(len(df.columns.tolist()))

scaler = MinMaxScaler()
X = scaler.fit_transform(df.values)

# Use GA to find best EIF parameters
#best_params = ga_eif(X, pop_size=10, generations=10)
#print(best_params)

forest = iForest(X=X,
                 ntrees=config.N_TREES,
                 sample_size=config.SAMPLE_SIZE,
                 ExtensionLevel=4)


scores_train = forest.compute_paths(X)
X_augmentated = np.hstack((X, scores_train.reshape(-1, 1)))

oc_svm_augmented = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
oc_svm_augmented.fit(X.reshape(-1, 1))

with open(config.EIF_PATH, "wb") as f:
    pickle.dump(forest, f)

with open(config.OC_SVM_PATH, "wb") as f:
    pickle.dump(oc_svm_augmented, f)

with open(config.SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)