import torch

class TextGenerator:
    def __init__(self, model, itos, stoi, context_size):
        self.model = model
        self.itos = itos
        self.stoi = stoi
        self.context_size = context_size
        
    def generate(self, context, max_len=10, temperature=1.0):
        """Generate text from initial context"""
        context = context[-self.context_size:]  # Truncate if needed
        context = [self.stoi.get(w, self.stoi['<pad>']) for w in context]
        context += [self.stoi['<pad>']] * (self.context_size - len(context))
        
        generated = []
        with torch.no_grad():
            for _ in range(max_len):
                x = torch.tensor(context).unsqueeze(0)
                logits = self.model(x.to(self.model.device))
                probs = torch.softmax(logits / temperature, dim=-1)
                next_idx = torch.multinomial(probs[0], 1).item()
                next_word = self.itos[next_idx]
                
                if next_word == '.':
                    break
                    
                generated.append(next_word)
                context = context[1:] + [next_idx]
                
        return ' '.join(generated)
