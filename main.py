from config import *
from data_processor import TextProcessor
from dataset import SequenceDataset
from model import NextWordPredictor
from trainer import LanguageModelTrainer
from generator import TextGenerator
from visualize import EmbeddingVisualizer

if __name__ == "__main__":
    # 1. Data preparation
    processor = TextProcessor(DATA_PATH)
    words = processor.load_and_preprocess()
    dataset = SequenceDataset(words, processor.stoi, CONTEXT_SIZE)
    
    # 2. Model setup
    model = NextWordPredictor(
        vocab_size=len(processor.stoi),
        emb_dim=EMB_DIM,
        hidden_size=HIDDEN_SIZE,
        context_size=CONTEXT_SIZE
    )
    
    # 3. Training
    trainer = LanguageModelTrainer(model, DEVICE)
    trainer.train(dataset, EPOCHS, BATCH_SIZE)
    trainer.save_model(MODEL_PATH)
    
    # 4. Text generation
    trainer.load_model(MODEL_PATH)
    generator = TextGenerator(model, processor.itos, processor.stoi, CONTEXT_SIZE)
    
    print("Sample generation:", generator.generate(["first", "citizen"], max_len=15))
    
    # 5. Embedding visualization
    visualizer = EmbeddingVisualizer(model)
    visualizer.visualize(processor.itos, "embeddings.png")
