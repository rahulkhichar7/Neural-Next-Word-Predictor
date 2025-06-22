from torch import optim
import torch

class LanguageModelTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
    def train(self, dataset, epochs, batch_size, lr=0.01):
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}')
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
