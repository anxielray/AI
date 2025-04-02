# Model Implementation

- This is where I define the architecture of my GPT-like model using PyTorch.
- I'll implement the core components of the Transformer decoder:
  - Self--Attention Mechanism
  - Feedforward Neural Networks
  - Layer Normalization
  - Residual Connections
  - Embedding Layer
  - Linear Layer + Softmax

## The Transformer Architecture

- The Transformer model consists of two main parts:

  - **Encoder:** Processes the input sequence into a contextualized representation (used in tasks like translation).

  - **Decoder:** Generates output step-by-step using attention mechanisms to focus on both previous outputs and encoded information.

- The decoder is the focus of autoregressive models like GPT (Generative Pre-trained Transformer).

### Structure of the Transformer

- Each Transformer decoder layer consists of:
  
#### 1. Masked Multi-Head Self--Attention
  
- Allows the decoder to attend to previous words in the sequence without looking ahead (future words are masked). 

- This is achieved using a triangular mask, ensuring that at position t, the model only "sees" words up to t-1.

#### 2. Multi-Head Cross-Attention (Optional)

- If the decoder is used in a seq-to-seq model (like translation), it attends to encoder outputs. - This allows it to understand the input sequence while generating the output.

#### 3. Feed--Forward Network (FFN)

- Applies two fully connected layers with a ReLU activation in between. - Helps in non-linearity and feature transformation.

#### 4. Layer Normalization & Residual Connections

- Helps stabilize training and allows gradients to flow properly.

#### 5. Positional Encoding

- Since Transformers lack recurrence, positional encodings are added to maintain word order information.

### How the Transformer Decoder Works

- When generating text, the decoder operates autoregressively, meaning it generates one token at a time while using previous tokens as input.

#### Step-by-Step Example for Text Generation

- Let’s say we want to generate "Hello, world!" one word at a time.

  - Input Token → Hello
  - Self-Attention → Hello attends to itself
  - Cross-Attention (if applicable) → Uses encoder context (for translation)
  - Feed-Forward → Generates probability distribution over next word
  - Prediction → Next word = ","
  - Repeat for next token → "world"...

- Each new word is generated based on previously generated words, without seeing future words
