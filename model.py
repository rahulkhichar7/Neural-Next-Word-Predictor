from torch import nn
import torch

class NextWordPredictor(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, context_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = nn.Linear(context_size * emb_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)  # Flatten context embeddings
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
