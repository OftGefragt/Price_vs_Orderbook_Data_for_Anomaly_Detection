EMBEDDING_DIM = 256 # dimension of the GRU embedding
TOP_K = 3 # top 3 anomalies
TIME_STEP = 30 # time steps for the input sequence
PERCENTILE = 95 # percentile for anomaly threshold

TRAIN_EPOCHS = 4 # number of training epochs (more have proven to worse results)
BATCH_SIZE = 64 # batch size for training
LEARNING_RATE = 1e-3 # learning rate for GRU optimizer
LEARNING_RATE_CENTER = 0.001 # learning rate for center update
N_LAYERS = 2 # number of GRU layers
DROPOUT_RATE = 0.2 # dropout rate for GRU layers
DEVICE = 'cpu' # device to use for training and inference

TRAIN_PATH = '../../data/train/BTC_1min.csv'
TEST_PATH = '../../data/test/ETH_1min.csv'
SVDD_PATH = '../models/svdd_center.pth'
GRU_ENCODER_PATH = '../models/gru_encoder.pth'