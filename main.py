# main.py
import sys

sys.path.append("./src")
from data_processing import process_data
from model_implementation import model, train
import torch

# Load data and create DataLoaders
print("Loading data...")  # Add this line
vocab_size, block_size, train_dataloader, val_dataloader = process_data.load_data()
print("Data loaded successfully.")  # Add this line

# Hyperparameters
embedding_dim = 64
num_heads = 4
num_layers = 3
dropout = 0.1
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
batch_size = 32
gradient_clip = 1.0

# Instantiate the model
print("Instantiating model...")  # Add this line
device = "cuda" if torch.cuda.is_available() else "cpu"
gpt_model = model.GPTModel(
    vocab_size, embedding_dim, num_heads, num_layers, block_size, dropout
).to(device)
print("Model instantiated successfully.")  # Add this line
print(gpt_model)

# Train the model
print("Starting training...")  # Add this line
train.train(
    gpt_model,
    train_dataloader,
    val_dataloader,
    device,
    vocab_size,
    max_iters,
    eval_interval,
    learning_rate,
    eval_iters,
    batch_size,
    gradient_clip,
)
print("Training completed.")  # Add this line
