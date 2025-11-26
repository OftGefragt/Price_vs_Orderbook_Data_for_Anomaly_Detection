TOP_K = 3 # top 3 anomalies
THRESHOLD = 0.95 # anomaly threshold

N_TREES = 235 # trees calculated by the GAN
SAMPLE_SIZE = 1691 # sample size calculated by the GAN

TRAIN_PATH = '../../data/train/BTC_1min.csv'
TEST_PATH = '../../data/test/ETH_1min.csv'
EIF_PATH = '../models/eif_model.pkl'
OC_SVM_PATH = '../models/oc_svm_augmented.pkl'
SCALER_PATH = '../models/scaler.pkl'
