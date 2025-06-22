import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class EmbeddingVisualizer:
    def __init__(self, model):
        self.model = model
        
    def visualize(self, itos, filename=None):
        embeddings = self.model.embedding.weight.data.cpu().numpy()
        tsne = TSNE(n_components=2)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(15, 15))
        for i in range(len(itos)):
            if i % 50 == 0:  # Plot subset for clarity
                x, y = embeddings_2d[i]
                plt.scatter(x, y)
                plt.text(x+0.1, y+0.1, itos[i], fontsize=8)
                
        plt.title("Word Embedding Visualization")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.grid(alpha=0.2)
        if filename:
            plt.savefig(filename)
        plt.show()
