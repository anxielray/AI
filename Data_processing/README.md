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
