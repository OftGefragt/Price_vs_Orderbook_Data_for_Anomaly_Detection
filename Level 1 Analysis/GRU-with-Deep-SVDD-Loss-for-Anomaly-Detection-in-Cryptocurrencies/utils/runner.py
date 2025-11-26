import torch
from utils.gru import GRUEncoder
from utils.inference import execute_inference
from utils.data_preparation import create_test_sequence
import config
# loading test data
X_test, test_data = create_test_sequence(config.TEST_PATH, time_step=10, load_scaler=False)

input_dim = X_test.shape[2]  # number of features per timestep
# loading trained GRU encoder
gru_encoder = GRUEncoder(input_dim=input_dim,embedding_dim=config.EMBEDDING_DIM,n_layers=config.N_LAYERS, dropout=config.DROPOUT_RATE)
gru_encoder.load_state_dict(torch.load(config.GRU_ENCODER_PATH))
# loading center c
c = torch.load(config.SVDD_PATH)

#getting the scores
execute_inference(gru_encoder, c, X_test, test_data, config.TIME_STEP, config.TOP_K, config.PERCENTILE)
