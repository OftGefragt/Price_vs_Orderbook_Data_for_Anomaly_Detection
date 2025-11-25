from utils.gru import GRUEncoder, train_deep_svdd
from utils.inference import execute_inference
from utils.data_preparation import create_train_test_sequences
import torch
import config

X_train, X_test, test_data = create_train_test_sequences(config.TRAIN_PATH, split_ratio=0.8, time_step=config.TIME_STEP)
print(X_train.shape)
print(X_test.shape)

input_dim = X_train.shape[2]  # number of features
embedding_dim = 256

#using a large embedding dim can help you reduce bias
gru_encoder = GRUEncoder(input_dim, embedding_dim, config.N_LAYERS, config.DROPOUT_RATE)

#We get the optimum c and save it.
c = train_deep_svdd(gru_encoder, X_train, config.TRAIN_EPOCHS, config.BATCH_SIZE, config.LEARNING_RATE, config.LEARNING_RATE_CENTER, config.DEVICE)

# Save center and GRU encoder weights
torch.save(c, config.SVDD_PATH)
torch.save(gru_encoder.state_dict(), config.GRU_ENCODER_PATH)

#getting the scores
execute_inference(gru_encoder, c, X_test, test_data, config.TIME_STEP, config.TOP_K, config.PERCENTILE)