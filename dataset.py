import torch

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, words, stoi, context_size=5):
        self.context_size = context_size
        self.stoi = stoi
        self.X, self.Y = self._create_sequences(words)
        
    def _create_sequences(self, words):
        """Create context-target pairs from continuous word sequence"""
        X, Y = [], []
        context = [self.stoi['<pad>']] * self.context_size
        
        for word in words:
            # Convert word to index, use <pad> if not found
            idx = self.stoi.get(word, self.stoi['<pad>'])
            X.append(context.copy())
            Y.append(idx)
            context = context[1:] + [idx]
            
        return torch.tensor(X), torch.tensor(Y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
