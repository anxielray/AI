# Data Processing

- This is the documentation of the Data processing session of the training of the model.
- Since we are starting with a small AI, small capabilities I started with a small dataset and then after training the model on small dataset, I will proceed to the big dataset.
- The dataset choice will be the famous `Tiny Shake Spear` dataset.
- After this, we want to clean the data to avoid giving our model lots of data, cause, remember, small capabilities.

## Cleaning the Data

- First I will reduce the dataset into lowercase, to reduce the consideration of `The` , `THE` and `the` and only consider the later.

## Vocabulary

- I'll create a mapping between each character in the lowercase TinyShakespeare text and a unique integer ID. This mapping will be used to convert the text into numerical data that the model can process.

## Encoding and Decoding

- Encode the whole dataset into a pytorch tensor.
- At this point we are going to spare some encoded data for validation to avoid overfitting(about 15%)

## Data Loading

- From the tensor, I am going to create dataloads that I will sequentially feed to the model for training
- The data is reloaded while loading and encoding it this time.
- The data is split into training-validation sets using the 90/10 split.
- This is done before creating a dataset, to ensure that no data leakages are enountered between validation data and training data.
- And speaking about data leaks, what do I mean?
  - Data leakage happens when data from outside the training dataset is used to create the model, but this future data will not be available when the model is used for prediction. The model will perform well in testing and validation, but when used in production, it becomes entirely inaccurate.
- I will create a Dataset class inherited from `torch.utils.data.Dataset` and use that to create  2 classes for the training set and the validation set.
- Since I just used an inherited dataset, we have to explicitly define 3 methods for the dataset object to emulate the Pytorch dataset object.
  - `__init__`: This method will initialize the dataset, taking the raw dataset and the blocksize of the sequence lenght, which is essentail for the proper working of the model.
  - `__len__`: This method will return the valid sequence that can be extracted.
  - `__getitem__`: The method will retrieve the sequence and the its target for training.
- The model is trained to predict the next character in a sequence. To do this, we will feed it an input sequence and its corresponding target character, that's the explanation for the `blocksize + 1` Meaning the blocksize shifted by one.
- The __len__ method in a PyTorch dataset returns the number of valid samples (sequences) that can be created from the dataset.
- If we have a dataset with N total characters and we need sequences of length block_size, we can't start a sequence from every character in data—we have to stop early to ensure we have enough characters for a full sequence.

### DataLoader creation

- I will create dataloaders for both the datasets.
- The shape of the the input and the target tensors; their shapes should be `[batch_size, block_size]`
- What are batch numbers?
  - Well batch numbers refer to the order of a batch during an epoch(In the context of training machine learning models, an "epoch" refers to one complete pass of the entire training dataset through the learning algorithm) in training.
  - A batch itself is the small samples of data we feed to the model while training it, to avoid feeding the whole dataset, say it was a big dataset it would consume a lot of memory If a dataset has 1000samples, and the batch number is 100, therefore the batch number/size will be equal to 10 per epoch.
  - Batch numbers are 0-index based.
