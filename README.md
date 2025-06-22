# Neural Next Word Predictor

A clean, modular deep learning project for next-word prediction using PyTorch. This repository enables you to train, evaluate, and visualize a neural network that predicts the next word in a text sequence. The project is structured for readability, extensibility, and ease of use.

GitHub: [Neural-Next-Word-Predictor](https://github.com/rahulkhichar7/Neural-Next-Word-Predictor.git)[1]

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Text Generation](#text-generation)
- [Embedding Visualization](#embedding-visualization)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Neural Next Word Predictor** leverages a feed-forward neural network to predict the next word in a given sequence of text. It is designed with modularity and clarity in mind, making it easy for anyone to understand, extend, and use for NLP research or educational purposes.

---

## Features

- **Modular Design**: Clean separation of data processing, modeling, training, generation, and visualization.
- **PyTorch Implementation**: Uses PyTorch for flexibility and performance.
- **Customizable**: Easily adjust hyperparameters and model architecture.
- **Visualization**: Visualize learned word embeddings using t-SNE.
- **Reproducible**: Centralized configuration and requirements.

---

## Project Structure

```
Neural-Next-Word-Predictor/
├── config.py               # Hyperparameters and settings
├── data_processor.py       # Text preprocessing and vocabulary
├── dataset.py              # PyTorch Dataset class
├── model.py                # Neural network model
├── trainer.py              # Training utilities
├── generator.py            # Text generation utilities
├── visualize.py            # Embedding visualization
├── main.py                 # Main execution script
├── requirements.txt        # Python dependencies
└── dataset.txt             # (Your text corpus)
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rahulkhichar7/Neural-Next-Word-Predictor.git
   cd Neural-Next-Word-Predictor
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Prepare your dataset:**  
   Place your training text in `dataset.txt` (or update the path in `config.py`).

2. **Train the model:**  
   Run the main script to preprocess data, train the model, and save the trained weights:
   ```bash
   python main.py
   ```

3. **Generate text:**  
   After training, the script will print a sample generated sequence to the console.

4. **Visualize embeddings:**  
   The script will also create a t-SNE visualization of the learned word embeddings and save it as `embeddings.png`.

---

## Configuration

All hyperparameters (context size, embedding dimension, hidden size, batch size, epochs, device, file paths) are defined in `config.py`.  
Adjust these values as needed to experiment with different setups.

---

## Training

- The model is trained to predict the next word given a fixed-length context window.
- Training progress and loss are printed every 10 epochs.
- The trained model is saved to disk for future use or inference.

---

## Text Generation

After training, you can generate text by providing an initial context.  
The `TextGenerator` class samples the next word iteratively, building a sequence until an end-of-sequence token is produced or the maximum length is reached.

Example output:
```
Sample generation: citizen of india and to
```

---

## Embedding Visualization

The `EmbeddingVisualizer` module projects learned word embeddings into 2D space using t-SNE and saves the plot as `embeddings.png`.  
This helps you explore how the model clusters semantically related words.

---

## Dataset

- The default dataset file is `dataset.txt`.
- You can use any plain text file. The text is cleaned and tokenized automatically.
- For best results, use a large and diverse corpus.

---

## Contributing

Contributions, suggestions, and bug reports are welcome!  
Feel free to fork the repository and submit a pull request.

---

## License

This project is licensed under the MIT License.

---

**Contact:**  
For questions or feedback, open an issue on [GitHub](https://github.com/rahulkhichar7/Neural-Next-Word-Predictor.git)[1].
