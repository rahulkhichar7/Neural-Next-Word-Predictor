import torch

# Hyperparameters and settings
CONTEXT_SIZE = 5
EMB_DIM = 32
HIDDEN_SIZE = 1024
BATCH_SIZE = 1024
EPOCHS = 800
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "next_word_predictor.pth"
DATA_PATH = "dataset.txt"