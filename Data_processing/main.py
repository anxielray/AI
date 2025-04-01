import torch
import os

# 1.1 Download and Load TinyShakespeare (if you don't have it already)

# This part downloads the data only if it does not exist already.
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
DATA_FILE = 'input.txt'
DATA_DIR = 'data'  # Create a directory to store the data

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

# 1.3 Lowercase the Text
text = text.lower()

# 1.2 Character-Level Tokenization and 1.4 Create Vocabulary
chars = sorted(list(set(text)))  # Get unique characters and sort them
vocab_size = len(chars)
print(f"Vocabulary Size: {vocab_size}") # print the vocab size so we know how many tokens we will need

# Create mappings from character to index and index to character
char_to_index = {ch: i for i, ch in enumerate(chars)}
index_to_char = {i: ch for i, ch in enumerate(chars)}


# Example usage:
print("Example:")
print(f"Text (first 100 chars): {text[:100]}")
print(f"Character-to-Index mapping (first 5): {list(char_to_index.items())[:5]}")
print(f"Index-to-Character mapping (first 5): {list(index_to_char.items())[:5]}")


# Function to encode text into a list of indices
def encode(text):
    return [char_to_index[ch] for ch in text]

# Function to decode a list of indices back into text
def decode(indices):
    return ''.join([index_to_char[i] for i in indices])

# Example Encode/Decode
encoded_text = encode("hello")
decoded_text = decode(encoded_text)
print(f"Encoded 'hello': {encoded_text}")
print(f"Decoded back: {decoded_text}")



# Convert the entire text to indices
data = torch.tensor(encode(text), dtype=torch.long) # Create a tensor from the encoded text
print(f"Length of encoded data: {len(data)}")
print(f"First 100 encoded values: {data[:100]}")

# Now, 'data' is a PyTorch tensor containing the encoded text.  This is what we'll feed into our model.


