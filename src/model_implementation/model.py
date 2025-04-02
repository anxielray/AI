import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters (these could also be defined in a config file or passed as arguments)
# embedding_dim = 64  # Size of the token embeddings
# num_heads = 4        # Number of attention heads (embedding_dim must be divisible by this)
# num_layers = 3       # Number of transformer decoder layers
# dropout = 0.1       # Dropout probability


# 1. Self-Attention
class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = (
            embedding_dim // num_heads
        )  # Ensure embedding_dim is divisible by num_heads

        assert (
            self.head_dim * num_heads == embedding_dim
        ), "embedding_dim must be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries, mask):
        # Get number of training examples
        N = queries.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split embedding into self.num_heads pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.values(values)  # (N, value_len, num_heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, num_heads, head_dim)
        queries = self.queries(queries)  # (N, query_len, num_heads, head_dim)

        # Scaled dot-product attention
        # Einsum is a compact way to express tensor operations
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, num_heads, head_dim), keys shape: (N, key_len, num_heads, head_dim)
        # energy shape: (N, num_heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embedding_dim ** (1 / 2)), dim=3)
        # attention shape: (N, num_heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_dim
        )
        # attention shape: (N, num_heads, query_len, key_len), values shape: (N, value_len, num_heads, head_dim)
        # out after einsum: (N, query_len, num_heads, head_dim), then combine last two dimensions

        # Linear layer to send it to embedding_dim size
        out = self.fc_out(out)
        # (N, query_len, embedding_dim)

        return self.dropout(out)  # Apply dropout here


# 2. Transformer Block (Decoder Layer)
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size, dropout):
        super().__init__()
        self.attention = SelfAttention(embedding_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.attention(x, x, x, mask)
        x = self.dropout(
            self.norm1(attention + x)
        )  # Residual connection and layer norm
        forward = self.feed_forward(x)
        x = self.dropout(self.norm2(forward + x))  # Residual connection and layer norm
        return x


# 3. GPT-like Model
class GPTModel(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, num_heads, num_layers, block_size, dropout
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(block_size, embedding_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, num_heads, block_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(
            embedding_dim
        )  # Layer norm before the final linear layer
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size  # Store block_size as an attribute

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).to(x.device)  # (seq_len,)
        positions = positions.unsqueeze(0).expand(N, seq_len)  # (N, seq_len)
        x = self.embedding(x) + self.position_embeddings(
            positions
        )  # (N, seq_len, embedding_dim)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)  # Apply layer norm before final linear layer
        x = self.fc(x)  # (N, seq_len, vocab_size)
        return x

    def generate(self, idx, max_new_tokens, device):
        """
        Generates a sequence of tokens of length up to max_new_tokens,
        conditional on the model and the initial sequence of tokens in the index idx
        """
        # idx is (B, T) array of indices in the current context
        block_size = self.block_size
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            mask = torch.tril(
                torch.ones(1, idx_cond.size(1), idx_cond.size(1), device=device)
            ).bool()
            logits = self(idx_cond, mask)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
