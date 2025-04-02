import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import os


# Function to estimate loss
# @torch.no_grad()
# def estimate_loss(model, train_dataloader, val_dataloader, device, eval_iters):
#     model.eval()
#     train_losses = torch.zeros(eval_iters)
#     val_losses = torch.zeros(eval_iters)
# for k in range(eval_iters):
#     X, Y = next(iter(train_dataloader))
#     X, Y = X.to(device), Y.to(device)
#     mask = torch.tril(torch.ones(1, X.size(1), X.size(1), device=device)).bool()
#     logits = model(X, mask)
#     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
#     train_losses[k] = loss.item()


#     Xv, Yv = next(iter(val_dataloader))
#     Xv, Yv = Xv.to(device), Yv.to(device)
#     maskv = torch.tril(torch.ones(1, Xv.size(1), Xv.size(1), device=device)).bool()
#     logitsv = model(Xv, maskv)
#     lossv = F.cross_entropy(logitsv.view(-1, logitsv.size(-1)), Yv.view(-1))
#     val_losses[k] = lossv.item()
# model.train()
# return train_losses.mean(), val_losses.mean()
@torch.no_grad()
def estimate_loss(model, train_dataloader, val_dataloader, device, eval_iters):
    model.eval()
    train_losses = torch.zeros(eval_iters)
    val_losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        # Create new iterators for each evaluation step
        train_dataloader_iter = iter(train_dataloader)
        val_dataloader_iter = iter(val_dataloader)

        try:
            X, Y = next(train_dataloader_iter)
            X, Y = X.to(device), Y.to(device)
            mask = torch.tril(torch.ones(1, X.size(1), X.size(1), device=device)).bool()
            logits = model(X, mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            train_losses[k] = loss.item()
        except StopIteration:
            print("Train dataloader exhausted during evaluation.")
            break

        try:
            Xv, Yv = next(val_dataloader_iter)
            Xv, Yv = Xv.to(device), Yv.to(device)
            maskv = torch.tril(
                torch.ones(1, Xv.size(1), Xv.size(1), device=device)
            ).bool()
            logitsv = model(Xv, maskv)
            lossv = F.cross_entropy(logitsv.view(-1, logitsv.size(-1)), Yv.view(-1))
            val_losses[k] = lossv.item()
        except StopIteration:
            print("Val dataloader exhausted during evaluation.")
            break
    model.train()
    return train_losses.mean(), val_losses.mean()


def train(
    model,
    train_dataloader,
    val_dataloader,
    device,
    vocab_size,
    max_iters=5000,
    eval_interval=500,
    learning_rate=3e-4,
    eval_iters=200,
    batch_size=32,
    gradient_clip=1.0,
):  # Added gradient_clip
    """
    Trains the GPT model.

    Args:
        model: The GPTModel instance.
        train_dataloader: DataLoader for the training set.
        val_dataloader: DataLoader for the validation set.
        device: 'cuda' or 'cpu'.
        vocab_size: Vocabulary Size
        max_iters: Number of training iterations.
        eval_interval: Evaluate validation loss every eval_interval iterations.
        learning_rate`: Learning rate for the optimizer.
        eval_iters: Number of iterations to average for the validation loss.
        batch_size: Batch size
        gradient_clip: Clip the gradients to this value
    """

    model = model.to(device)  # Ensure model is on the correct device

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    start_time = time.time()
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            train_loss, val_loss = estimate_loss(
                model, train_dataloader, val_dataloader, device, eval_iters
            )
            print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        # Sample a batch of data
        try:
            batch = next(iter(train_dataloader))
        except StopIteration:
            # If the iterator is exhausted, re-initialize it
            train_dataloader_iter = iter(train_dataloader)
            batch = next(train_dataloader_iter)

        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        # Forward pass
        N, seq_len = inputs.shape
        mask = torch.tril(torch.ones((N, seq_len, seq_len), device=device)).bool()
        outputs = model(inputs, mask)  # (N, seq_len, vocab_size)

        # Calculate Loss
        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping (Crucial for Transformers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

    end_time = time.time()
    print(f"Training finished. Total time: {end_time - start_time:.2f} seconds")

    # Save the trained model
    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), "./models/model.pth")
    print("model saved")
