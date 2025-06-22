import re

class TextProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.stoi = {}
        self.itos = {}
        self.vocab = set()
        
    def load_and_preprocess(self):
        """Load and preprocess entire corpus as single sequence"""
        with open(self.file_path, 'r') as f:
            text = f.read()
        
        # Clean text (keep alphanumeric, space, and period)
        text = re.sub(r'[^a-zA-Z0-9 .]', '', text).lower()
        
        # Split into words while preserving periods as tokens
        words = text.split()
        words.append('.')  # Add final EOS marker
        
        # Build vocabulary
        self.vocab = sorted(set(words))
        self.stoi = {word: i for i, word in enumerate(self.vocab)}
        self.stoi['<pad>'] = len(self.vocab)  # Padding token
        self.itos = {i: word for word, i in self.stoi.items()}
        
        return words
