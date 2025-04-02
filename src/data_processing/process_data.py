import torch
import os
from torch.utils.data import Dataset, DataLoader

def load_data():  # DEFINE THE load_data FUNCTION
    # 1. Re-run Data Loading and Encoding (from previous step) - For completeness
    DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    DATA_FILE = 'input.txt'
    DATA_DIR = 'data'

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(os.path.join(DATA_DIR, DATA_FILE)):
        import urllib.request
        print("Downloading Tiny Shakespeare dataset...")
        urllib.request.urlretrieve(DATA_URL, os.path.join(DATA_DIR, DATA_FILE))
        print("Dataset downloaded.")

    filepath = os.path.join(DATA_DIR, DATA_FILE)
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    text = text.lower()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_index = {ch: i for i, ch in enumerate(chars)}
    index_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(text):
        return [char_to_index[ch] for ch in text]

    def decode(indices):
        return ''.join([index_to_char[i] for i in indices])

    data = torch.tensor(encode(text), dtype=torch.long) # CREATE TENSOR HERE

    # 2. Create Training and Validation Sets
    train_val_split = 0.9  # 90% for training, 10% for validation
    n = int(train_val_split * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Length of training data: {len(train_data)}")
    print(f"Length of validation data: {len(val_data)}")


    # 3. Define a Custom Dataset
    class CharDataset(Dataset):
        # def __init__(self, data, block_size):
        #     self.data = data
        #     self.block_size = block_size #sequence_length from last turn

        # def __len__(self):
        #     return len(self.data) - self.block_size #ensure that we always have a full block_size sequence

        # def __getitem__(self, idx):
        #     # Grab a chunk of (block_size + 1) characters from the data
        #     chunk = self.data[idx:idx + self.block_size + 1] # + 1 because we need input AND target
        #     x = chunk[:-1].clone()class CharDataset(Dataset):
        def __init__(self, data, block_size):
            self.data = data
            self.block_size = block_size #sequence_length from last turn

        def __len__(self):
            return len(self.data) - self.block_size #ensure that we always have a full block_size sequence

        def __getitem__(self, idx):
            # Grab a chunk of (block_size + 1) characters from the data
            chunk = self.data[idx:idx + self.block_size + 1] # + 1 because we need input AND target
            x = chunk[:-1].clone() # Input sequence (all but the last token)
            y = chunk[1:].clone()  # Target sequence (all but the first token, shifted by one)

            return {'input': x, 'target': y} # Return as a dictionary
 # Input sequence (all but the last token)
            y = chunk[1:].clone()  # Target sequence (all but the first token, shifted by one)

            return {'input': x, 'target': y} # Return as a dictionary

    # 4. Create DataLoaders
    block_size = 32 #sequence_length from last turn

    train_dataset = CharDataset(train_data, block_size)
    val_dataset = CharDataset(val_data, block_size)

    batch_size = 32  # You can adjust this
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  #shuffle the training set to reduce overfitting
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)   #we do not shuffle the validation set, we want consistent output

    # 5. Test the DataLoader (Important!)
    # Get a batch from the training dataloader
    batch = next(iter(train_dataloader))
    inputs = batch['input']
    targets = batch['target']

    print("Sample Batch:")
    print(f"Input shape: {inputs.shape}")    # Should be [batch_size, block_size]
    print(f"Target shape: {targets.shape}")  # Should be [batch_size, block_size]
    print(f"Sample input batch:\n{inputs}")
    print(f"Sample target batch:\n{targets}")

    # Example: Decode a sequence from the batch
    sample_sequence_index = 0
    sample_input = inputs[sample_sequence_index].tolist()
    sample_target = targets[sample_sequence_index].tolist()
    print(f"Decoded input sequence: {decode(sample_input)}")
    print(f"Decoded target sequence: {decode(sample_target)}")

    return vocab_size, block_size, train_dataloader, val_dataloader  # RETURN THE VALUES

