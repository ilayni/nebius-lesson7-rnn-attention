# Homework — NLP & Sequence Modeling

In this assignment, you will build end-to-end NLP pipelines and sequence models, combining classical text processing techniques with neural networks.

The homework is divided into two main parts:

1. Part 1 — Text Classification (AG News) :

    In this part, you will implement a full pipeline for text classification — from raw text to predictions.

    What you will learn:
    * How text is converted into tokens (subword tokenization)
    * How to learn word representations (embeddings)
    * How to use embeddings as input to neural networks
    * How to evaluate and analyze model performance

2. Part 2 — Character-Level Language Model (Char-RNN)

    In this part, you will build a sequence model that learns to generate text at the character level.

    What you will learn:
    * How sequence data is represented
    * How RNNs process sequential information
    * How generative models produce new sequences

**Submission Requirements (Important)**

Please make sure that:

1. All tasks are fully implemented
2. All questions are answered clearly
3. All notebook cells are executed
4. Do NOT submit notebooks with cleared outputs
5. Do NOT leave incomplete or placeholder code

Your submission should allow us to:

See outputs, results, and analysis

**Goal of the Assignment**

By completing this homework, you will:

* Build a complete NLP pipeline from scratch
* Understand how embeddings are learned and used
* Gain hands-on experience with neural networks for text
* Develop intuition for sequence modeling and generation

## Part 1- AG News with BPE + Word2Vec + MLP - 50 points

For this task we will use the [AG News Dataset](https://huggingface.co/datasets/sh0416/ag_news).


```python
#Load the Data
from datasets import load_dataset

dataset = load_dataset("ag_news")
label_names = dataset["train"].features["label"].names

train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

print(train_df.shape, test_df.shape)
train_df.head()
```

    (120000, 2) (7600, 2)






  <div id="df-596ceb15-6583-42ad-bb77-248feca0b7ce" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wall St. Bears Claw Back Into the Black (Reute...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Carlyle Looks Toward Commercial Aerospace (Reu...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Oil and Economy Cloud Stocks' Outlook (Reuters...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Iraq Halts Oil Exports from Main Southern Pipe...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Oil prices soar to all-time record, posing new...</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-596ceb15-6583-42ad-bb77-248feca0b7ce')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-596ceb15-6583-42ad-bb77-248feca0b7ce button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-596ceb15-6583-42ad-bb77-248feca0b7ce');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




### **Task 1 - Tokenization (BPE or WordPiece) - 10 points**

**Goal**

Convert raw text into tokens using subword tokenization.

Subword tokenization helps models handle:

* rare words

* unknown words

* morphological variation

          
|Example sentence | Word-level tokenization |Subword tokenization (BPE example) |
|----------------|:--------------------|:------------------|
|"unbelievable movie" |["unbelievable", "movie"]|   ["un", "believ", "able", "movie"]|



**Task**

Implement tokenization using Byte Pair Encoding (BPE) and WordPiece.


Read more here:

* [tokenizers](https://huggingface.co/docs/tokenizers/quicktour)


Train the tokenizer on the training corpus only.

Choose a vocabulary size (e.g., 10k tokens).

**Deliverables**

write 5 sentences and run through each tokenizer.
for example:

|Sentence|Tokens [BPE] | Tokens [WordPiece] |
|--------|:------|----------------------|
|this movie was unbelievable |["this", "movie", "was", "un", "believ", "able"]|["this", "movie", "was", "un", "##bel", "##ie", "##va", "##ble"]|

           

**Question**

What is the difference between BPE and WordPiece tokenization methods? When would you prefer to use each? Why both is considered better than wordlevel?


```python
#code goes here

#sentences for example you may write your own

sentences = [
    "this movie was unbelievable",
    "the smartphone includes ultra-fast charging and long-lasting battery life",
    "scientists discovered a new microorganism in antarctica",
    "the football player scored twice in the championship match",
    "economic uncertainty affected international markets"
]
```

write your answers

### **Task 2 — Train Word Embeddings (Word2Vec) - 10 points**

**Goal**

Train Word2Vec embeddings to represent BPE tokens as dense vectors.

Instead of representing words as large sparse vectors (Bag-of-Words), embeddings map words to a low-dimensional vector space.

|**Example**|**Embedding**|
|-----------|:------------|
|king       |[0.32, -0.12, 0.88, ...]|
|queen      |[0.30, -0.10, 0.91, ...]|


Words with similar meanings should have similar vectors.

**Task**

1. Train two Word2Vec models on the BPE-tokenized training sentences:

    * using Skip-gram

    * using CBOW

    * using diffrent embedding size [50,100,200]

    You may use the gensim library.

2. Run 3 examples embeddings and most similar words for example:

    * Embedding("movie") = [0.12, -0.44, 0.21, ...]

    * similar words (model.wv.most_similar("movie")): film, cinema, story

3. Then compare the two models and explain:

    * Which model performed better for this task
    * Why it may be better on this dataset
    * What embedding dimension you chose and why ?
    * What vocabulary size was actually learned, and why it may differ from the tokenizer vocabulary size


```python
!pip install --upgrade gensim
```


```python
from gensim.models import Word2Vec
import numpy as np


tokenized_sentences = [
    # TODO: encode each text using bpe_tokenizer
]


embedding_sizes = [50, 100, 200]

model_configs = [
    {"name": "CBOW", "sg": 0},        # sg=0 → CBOW
    {"name": "Skip-gram", "sg": 1},   # sg=1 → Skip-gram
]

models = {}


for config in model_configs:
    for emb_dim in embedding_sizes:

        model_name = f"{config['name']}_{emb_dim}"

        # TODO: Initialize Word2Vec model
        model = Word2Vec(
            sentences=tokenized_sentences,
            vector_size=...,   # TODO
            window=...,        # TODO (hint: 5)
            min_count=...,     # TODO (hint: 2)
            sg=...,            # TODO
            workers=4
        )

        models[model_name] = model


example_tokens = ["movie", "sport", "market"]

for model_name, model in models.items():
    print("\n" + "="*60)
    print(f"Model: {model_name}")

    for token in example_tokens:

        if token not in model.wv:
            continue

        print(f"\nToken: {token}")

        # TODO: print embedding vector (first 10 values)

        # TODO: print most similar words
        # hint: model.wv.most_similar(token)
```

write your answers

### **Task 3- MLP (Multi-Layer Perceptron) Architecture and Training - 10 points**

After tokenizing the text and training the embedding model, we are ready for the classification stage.

In this part, you will use the learned token embeddings as input features for an MLP (Multi-Layer Perceptron) model.

An MLP is a simple feed-forward neural network made of fully connected layers and non-linear activations. Unlike Word2Vec, which learns vector representations of tokens based on context, the MLP learns to map these representations to the target labels. In this assignment, the goal is to classify each news text into its correct category.

![image.png]([IMAGE_STRIPPED])

Because the each input training text contains a variable number of tokens, **you must first convert each tokenized sentence into a fixed-size vector representation**. One simple approach is to aggregate the token embeddings in the sentence, for example by taking the mean of all token embeddings. This sentence-level vector can then be passed into the MLP.

**Task**

Design and train an MLP model for text classification on the AG News dataset.

The model should take as input a sentence representation built from the learned BPE token embeddings, and output a prediction over the class labels.

**Requirements**

1. Convert each tokenized sentence into a fixed-length vector using the trained embeddings.
You may use:

    * mean pooling
    * sum pooling
    * If you choose a different method, explain it.

2. Design an MLP for the classification task.

Your model should include:
    * an input layer matching the embedding dimension
    * at least one hidden layer and up to 3
    * a non-linear activation function such as ReLU
    * an output layer with the correct number of classes

3. Train the model on the training set.

4. Clearly state your design choices, including:
    * the input representation
    * number of hidden layers
    * hidden dimension(s)
    * activation function
    * optimizer
    * learning rate
    * batch size
    * number of epochs

4. Briefly explain why an MLP can be used here, and what its limitation is compared to sequence-aware models such as RNNs.

The pipeline is:

Text → BPE tokens → token embeddings → pooling → MLP → class probabilities



```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# 1. Tokenize the texts using the trained BPE tokenizer

train_tokens = ...
test_tokens  = ...


# 2. Choose your best trained Word2Vec model from task 2
w2v_model = ...
embedding_dim = w2v_model.wv.vector_size

print("Chosen embedding dimension:", embedding_dim)


# 3. Convert each sentence into ONE fixed-size vector

def sentence_to_vec(tokens, model, dim):
    vectors = []

    for tok in tokens:
        if tok in model.wv:
            vectors.append(model.wv[tok])

    # If no token in the sentence appears in the embedding vocabulary,
    # return a zero vector.
    if len(vectors) == 0:
        return np.zeros(dim, dtype=np.float32)

    # TODO: return mean pooling or sum pooling or max pooling

    return ...

# Build one fixed-size vector per text
X_train = np.array([sentence_to_vec(t, w2v_model, embedding_dim) for t in train_tokens], dtype=np.float32)
X_test  = np.array([sentence_to_vec(t, w2v_model, embedding_dim) for t in test_tokens], dtype=np.float32)

y_train = np.array(train_df["label"])
y_test  = np.array(test_df["label"])

print("X_train shape:", X_train.shape)   # expected: (num_train_samples, embedding_dim)
print("X_test shape:", X_test.shape)     # expected: (num_test_samples, embedding_dim)


# 4. Convert data into PyTorch Dataset / DataLoader

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TextDataset(X_train, y_train)
test_dataset  = TextDataset(X_test, y_test)

# TODO:
# Try changing batch_size and explain your choice
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 5. Define the MLP model

# TODO:
# Design your own MLP architecture.
# You may change:
# - number of hidden layers
# - hidden dimension
# - activation functions
# - dropout


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.model = ... # hint use nn.Sequential to build the models layers , nn.Linear for the linear layers and nn.ReLU for the activation layer
    def forward(self, x):
        return self.model(x)

# TODO:
# Choose hidden_dim and justify your choice
model =...


# 6. Training setup
#
# TODO:
# choose optimizer and learning rate

criterion = ...
optimizer = ...


# 7. Training loop

epochs = ...

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


```

write your answers

### **Task 4 - Evaluation and Error Analysis - 20 points**


After training your MLP model, evaluate its performance on the test set.

**Goals**
  * Measure classification performance using accuracy
  * Identify and analyze model errors
  * Better understand model limitations

**Requirements**
1. Compute test accuracy and confusion metrix
2. Print 3 examples where the model made incorrect predictions. For each error, display:
    * Original text
    * True label
    * Predicted label
3. Do you see a pattern or can point on where the model fails?
3. Suggest one method to analyze which parts of the input text influenced the model's prediction.
You do not need to implement it — only explain the idea.


```python
# code goes here
```

write your answers

## Part 2- Char-RNN Model - 50 points

Similarly to word-level language model, char-level RNN model processes one token at a time, but in this case a token represents a single character.


![dinos3.png]([IMAGE_STRIPPED])

In this part you will build a model that generating dinosaur names, the dataset can be found [here](https://gist.github.com/judahsemi/210d42e0410c122aa0e437861c46cf94).

This a txt file with dinosaur names. Your mission is to get creative and create new names of unknown dinosaurs.

### **Task 1 - Build a PyTorch Dataset for RNN Training - 10 points**

**Goal**

Create a custom PyTorch Dataset for training a character-level RNN model on a dataset of names.

**Input**

Your dataset should be initialized with:

1. seq_len - the length of each sequence
2. data - path to txt file or a list of names

**Requierments**
1. Preprocess the data
      * Add a start token '<' at the beginning of each name
      * Add an end token '>' at the end of each name
  
    Example: "anna" → "`<anna>`"
    
    Concatenate all processed names into one long string

2. Build vocabulary
      * Extract all unique characters from the data (including '<' and '>')
      * Create:
        - token_to_id mapping
        - id_to_token mapping

      * Store them in the dataset as:
        - self.vocab
        - self.token_to_id
        - self.id_to_token

3. Create sequences
      * Split the full text into sequences of length seq_len
      * Each dataset item should be based on a sliding window over the text.

4. Define dataset items
      * Each item should return a pair of tensors:

      * Input tensor (x) -
        sequence of character IDs of length seq_len
      * Target tensor (y) -
        same sequence shifted by one character

Example:
    
    Text:    "<anna>"
    seq_len: 4

    x: "<ann"
    y: "anna"

5. Implement Dataset methods

      * Your dataset class must implement:

        - __len__() → returns the number of sequences
        - __getitem__(idx) → returns (x, y) as PyTorch tensors



```python
import torch

class DinosDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len):
        self.seq_len = seq_len

        # TODO:
        # 1. Load data if data is a file path
        # 2. Add < and > to each name
        # 3. Concatenate all names into one string

        self.text = ...

        # TODO:
        # 4. Build vocabulary
        self.vocab = ...
        self.token_to_id = ...
        self.id_to_token = ...

        # TODO:
        # 5. Convert text to list of ids
        self.data = ...

    def __len__(self):
        # TODO:
        return ...

    def __getitem__(self, idx):
        # TODO:
        # extract sequence of length seq_len
        # create input x and target y (shifted by 1)
        x = ...
        y = ...
        return x, y
```

Split your dataset to get train and validation PyTorch datasets using method from `datasets` library.:


```python
# Your code here
```

Prepare PyTorch Dataloaders for train and validation datasetssets with fixed batch_size.


```python
# Your code here


```


```python
# example of a single batch from dataloader
batch = next(iter(train_loader))
batch
```




    [tensor([[ 5, 16, 14,  6, 26, 12, 16, 20,  2, 22],
             [26, 19, 16, 19,  2, 17, 21, 16, 19,  1],
             [26, 17, 21, 16, 17, 20,  1,  0, 21, 10],
             [22, 23, 10,  4, 22, 19, 20, 16, 19,  1],
             [ 2, 22, 19,  2,  1,  0, 20,  6, 19, 10],
             [14, 16, 15, 21, 16, 20,  2, 22, 19, 22],
             [20,  2, 22, 19, 22, 20,  1,  0,  6, 22],
             [19, 22, 20,  1,  0, 20,  9,  2, 15, 25],
             [ 2,  6, 16, 20,  2, 22, 19, 22, 20,  1],
             [22, 20,  1,  0,  2, 11,  2, 15,  4, 10],
             [20, 17, 20,  2, 22, 19, 22, 20,  1,  0],
             [ 2, 19,  4,  9,  2, 19, 10,  2,  1,  0],
             [13, 16, 20,  2, 22, 19, 22, 20,  1,  0],
             [16, 20,  2, 22, 19, 22, 20,  1,  0, 21],
             [15,  2, 21,  9, 22, 20,  1,  0, 13, 26],
             [ 0,  2,  4,  9,  6, 13, 16, 22, 20,  2],
             [ 0,  8, 16,  3, 10, 20,  2, 22, 19, 22],
             [21, 22, 20,  1,  0, 17,  2,  4,  9, 26],
             [22, 15,  8, 16, 20,  2, 22, 19, 22, 20],
             [ 3,  6, 10, 20,  2, 22, 19, 22, 20,  1],
             [16,  5, 16, 15,  1,  0,  7,  6, 15,  6],
             [20,  2, 22, 19, 22, 20,  1,  0, 14,  2],
             [22, 20,  1,  0,  2, 14, 17,  9, 10,  4],
             [ 0,  4, 19,  2, 20, 17,  6,  5, 16,  5],
             [26,  8,  5,  2, 13, 16,  5, 16, 15,  1],
             [ 0,  2, 13,  3,  6, 19, 21,  2, 23,  6],
             [ 2, 21, 10, 19,  9, 10, 15, 22, 20,  1],
             [16,  4, 22, 19, 20, 16, 19,  1,  0, 20],
             [19,  6, 22, 20,  1,  0,  6, 13, 16, 20],
             [15, 21, 19, 16,  5,  6, 14, 22, 20,  1],
             [17, 13,  2, 15, 10,  4, 16, 25,  2,  1],
             [15, 21,  6,  8, 10,  2,  1,  0,  6, 16],
             [19, 22, 20,  1,  0, 20, 10, 15, 16, 20],
             [ 4,  6, 19,  2, 21, 16, 17, 20,  1,  0],
             [25, 22,  2, 15,  9, 22,  2,  4,  6, 19],
             [ 3,  6, 13,  6, 20, 16,  5, 16, 15,  1],
             [26, 17, 21, 16, 20,  2, 22, 19, 22, 20],
             [ 0,  4,  6, 19,  2, 21, 16, 15, 26, 12],
             [16, 13, 16, 17,  9, 22, 20,  1,  0, 20],
             [20,  2, 22, 19, 22, 20,  1,  0,  8, 10],
             [ 2,  8, 19, 16, 20,  2, 22, 19, 22, 20],
             [20, 18, 22,  2, 14,  2,  1,  0,  9, 16],
             [ 1,  0,  5,  6, 10, 15, 16,  4,  9,  6],
             [16,  4, 22, 20,  1,  0, 20, 21,  6, 15],
             [ 2, 21,  2,  8, 16, 21, 10, 21,  2, 15],
             [10, 20,  2, 22, 19, 22, 20,  1,  0, 20],
             [10,  2, 15, 20,  2, 22, 19, 22, 20,  1],
             [10, 20,  2, 22, 19, 22, 20,  1,  0,  4],
             [ 0, 13,  2,  8,  6, 19, 17,  6, 21, 16],
             [ 8,  2, 21,  9, 16, 13, 22, 20,  1,  0],
             [ 2, 15, 16, 20,  2, 22, 19, 22, 20,  1],
             [20,  1,  0,  9,  6, 21,  6, 19, 16,  5],
             [ 2, 22, 19, 22, 20,  1,  0,  3, 16, 15],
             [ 0,  4, 16, 15,  4,  9, 16, 19,  2, 17],
             [ 0, 13, 16, 15,  8, 10, 20, 18, 22,  2],
             [27,  2, 15, 16, 24, 20, 12, 10, 20,  2],
             [19, 22, 20,  1,  0,  5,  2, 21, 16, 15],
             [ 0, 20, 21,  6, 15, 16, 17,  6, 13, 10],
             [ 1,  0, 20, 10, 15, 19,  2, 17, 21, 16],
             [22, 20,  1,  0,  5, 10,  2,  3, 13, 16],
             [ 1,  0,  3, 22, 19, 10, 16, 13,  6, 20],
             [ 9,  2, 16,  4,  9, 10, 13, 16, 15,  8],
             [15, 16, 20, 17,  9, 10, 21, 26, 20,  1],
             [ 5, 19, 26, 17, 21, 16, 20,  2, 22, 19]]),
     tensor([[16, 14,  6, 26, 12, 16, 20,  2, 22, 19],
             [19, 16, 19,  2, 17, 21, 16, 19,  1,  0],
             [17, 21, 16, 17, 20,  1,  0, 21, 10, 21],
             [23, 10,  4, 22, 19, 20, 16, 19,  1,  0],
             [22, 19,  2,  1,  0, 20,  6, 19, 10, 12],
             [16, 15, 21, 16, 20,  2, 22, 19, 22, 20],
             [ 2, 22, 19, 22, 20,  1,  0,  6, 22, 20],
             [22, 20,  1,  0, 20,  9,  2, 15, 25, 10],
             [ 6, 16, 20,  2, 22, 19, 22, 20,  1,  0],
             [20,  1,  0,  2, 11,  2, 15,  4, 10, 15],
             [17, 20,  2, 22, 19, 22, 20,  1,  0, 20],
             [19,  4,  9,  2, 19, 10,  2,  1,  0,  4],
             [16, 20,  2, 22, 19, 22, 20,  1,  0,  3],
             [20,  2, 22, 19, 22, 20,  1,  0, 21,  2],
             [ 2, 21,  9, 22, 20,  1,  0, 13, 26, 21],
             [ 2,  4,  9,  6, 13, 16, 22, 20,  2, 22],
             [ 8, 16,  3, 10, 20,  2, 22, 19, 22, 20],
             [22, 20,  1,  0, 17,  2,  4,  9, 26, 20],
             [15,  8, 16, 20,  2, 22, 19, 22, 20,  1],
             [ 6, 10, 20,  2, 22, 19, 22, 20,  1,  0],
             [ 5, 16, 15,  1,  0,  7,  6, 15,  6, 20],
             [ 2, 22, 19, 22, 20,  1,  0, 14,  2,  4],
             [20,  1,  0,  2, 14, 17,  9, 10,  4, 16],
             [ 4, 19,  2, 20, 17,  6,  5, 16,  5, 16],
             [ 8,  5,  2, 13, 16,  5, 16, 15,  1,  0],
             [ 2, 13,  3,  6, 19, 21,  2, 23,  6, 15],
             [21, 10, 19,  9, 10, 15, 22, 20,  1,  0],
             [ 4, 22, 19, 20, 16, 19,  1,  0, 20, 21],
             [ 6, 22, 20,  1,  0,  6, 13, 16, 20,  2],
             [21, 19, 16,  5,  6, 14, 22, 20,  1,  0],
             [13,  2, 15, 10,  4, 16, 25,  2,  1,  0],
             [21,  6,  8, 10,  2,  1,  0,  6, 16, 17],
             [22, 20,  1,  0, 20, 10, 15, 16, 20,  2],
             [ 6, 19,  2, 21, 16, 17, 20,  1,  0, 15],
             [22,  2, 15,  9, 22,  2,  4,  6, 19,  2],
             [ 6, 13,  6, 20, 16,  5, 16, 15,  1,  0],
             [17, 21, 16, 20,  2, 22, 19, 22, 20,  1],
             [ 4,  6, 19,  2, 21, 16, 15, 26, 12, 22],
             [13, 16, 17,  9, 22, 20,  1,  0, 20, 10],
             [ 2, 22, 19, 22, 20,  1,  0,  8, 10,  8],
             [ 8, 19, 16, 20,  2, 22, 19, 22, 20,  1],
             [18, 22,  2, 14,  2,  1,  0,  9, 16, 17],
             [ 0,  5,  6, 10, 15, 16,  4,  9,  6, 10],
             [ 4, 22, 20,  1,  0, 20, 21,  6, 15, 16],
             [21,  2,  8, 16, 21, 10, 21,  2, 15,  1],
             [20,  2, 22, 19, 22, 20,  1,  0, 20, 10],
             [ 2, 15, 20,  2, 22, 19, 22, 20,  1,  0],
             [20,  2, 22, 19, 22, 20,  1,  0,  4,  2],
             [13,  2,  8,  6, 19, 17,  6, 21, 16, 15],
             [ 2, 21,  9, 16, 13, 22, 20,  1,  0, 13],
             [15, 16, 20,  2, 22, 19, 22, 20,  1,  0],
             [ 1,  0,  9,  6, 21,  6, 19, 16,  5, 16],
             [22, 19, 22, 20,  1,  0,  3, 16, 15,  2],
             [ 4, 16, 15,  4,  9, 16, 19,  2, 17, 21],
             [13, 16, 15,  8, 10, 20, 18, 22,  2, 14],
             [ 2, 15, 16, 24, 20, 12, 10, 20,  2, 22],
             [22, 20,  1,  0,  5,  2, 21, 16, 15,  8],
             [20, 21,  6, 15, 16, 17,  6, 13, 10, 25],
             [ 0, 20, 10, 15, 19,  2, 17, 21, 16, 19],
             [20,  1,  0,  5, 10,  2,  3, 13, 16,  4],
             [ 0,  3, 22, 19, 10, 16, 13,  6, 20, 21],
             [ 2, 16,  4,  9, 10, 13, 16, 15,  8,  1],
             [16, 20, 17,  9, 10, 21, 26, 20,  1,  0],
             [19, 26, 17, 21, 16, 20,  2, 22, 19, 16]])]



### **Task 2 - Implement one-hot encoding - 5 points**

Instead of embedding layer, we will use one-hot encoding to pass input characters to LSTM layers of the network. It means that each character would be represented with a vector of dimension equal to the size of vocabulary, with a single non-zero element (the element equals 1) at the position of the character's index.

```
char = 'b'
id[char] = 2
one_hot = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
```

Write function `one_hot_encode` to implement one-hot encoding for a character index vector. It should take array containing sequences of character ids and vocab_size as arguments and return one-hot encoded numpy array of shape `(initial_array_shape, vocab_size)`.


```python
def one_hot_encode(array, vocab_size):
  one_hot_encoded = ...
  return one_hot_encoded
```


```python
# example of encoded batch
# batch[0] contains the first sequences in a batch
encoded = one_hot_encode(batch[0], len(train_data.vocab))
encoded
```




    array([[[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 1., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]],
    
           [[0., 0., 0., ..., 0., 1., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 1., 0., ..., 0., 0., 0.]],
    
           [[0., 0., 0., ..., 0., 1., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [1., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]],
    
           ...,
    
           [[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 1., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]],
    
           [[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 1., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 1., 0., ..., 0., 0., 0.]],
    
           [[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 1., 0.],
            ...,
            [0., 0., 1., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]]], dtype=float32)



### **Task 3 - Model - 5 points**

Here's a character level LSTM model. We don't use Embedding layer as it will be replaced with one-hot encoding during training.


```python
class CharRNN(nn.Module):

    def __init__(self, vocab_size, n_steps=100, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.vocab_size = vocab_size

        self.lstm = nn.LSTM(self.vocab_size, n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(n_hidden, self.vocab_size)


    def forward(self, x, hc):
        x, (h, c) = self.lstm(x, hc)
        x = self.dropout(x)

        # output of LSTM layer has shape [batch_size, sequence_length, hidden_dimension]
        # for fully connected layer we need to transform the output to the shape [batch_size * sequence_length, hidden_dimension]
        x = x.contiguous().view(x.size()[0]*x.size()[1], self.n_hidden)
        x = self.fc(x)

        return x, (h, c)

    def init_hidden(self, batch_size, device):
        ''' Initializes hidden state '''

        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))

```

Run one pass through the model to get predictions for a single batch.

For a fixed batch:
- apply one-hot encoding,
- run `forward` method of the model step by step, and print out the output of each layer and its dimensions,
- apply softmax to the output of forward pass to get probability distributions for a batch.


```python
# Your code here
```

### **Task 4 - Training - 10 points**

Now we can define a training function. Note that it calls `one_hot_encode` function that you've implemented in the previous task.


```python
def train(net, train_dataloader, val_dataloader, epochs=10, batch_size=64, n_steps=10, lr=0.001, clip=5,
          device='cpu', print_every=10):
    '''
        net: CharRNN model
        train_dataloader: train dataloader
        val_dataloader: validation dataloader
        epochs: number of epochs
        batch_size: batch size
        n_steps: number of character steps (sequence length defined in dataset)
        lr: learning rate
        clip: gradient clipping
        device: device_type (GPU or CPU)
        print_every: number of steps for printing loss
    '''

    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss().to(device)

    net.to(device)

    counter = 0
    n_chars = net.vocab_size

    for e in range(epochs):

        h = net.init_hidden(batch_size, device)

        for x, targets in train_dataloader:

            counter += 1

            # your function for one-hot encoding
            x = one_hot_encode(x, n_chars)
            inputs = torch.from_numpy(x)

            inputs.to(device)
            targets.to(device)

            # create new variables for the hidden state not to backprop through the entire training history
            h = tuple([each.data for each in h])

            net.zero_grad()

            output, h = net.forward(inputs, h)
            loss = criterion(output, targets.view(batch_size*n_steps))

            loss.backward()

            # clip_grad_norm helps to prevent the exploding gradients in RNN
            nn.utils.clip_grad_norm_(net.parameters(), clip)

            opt.step()

            if counter % print_every == 0:
                val_h = net.init_hidden(batch_size, device)
                val_losses = []

                for x, y in val_dataloader:
                    # your function for one-hot encoding
                    x = one_hot_encode(x, n_chars)
                    x = torch.from_numpy(x)

                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    inputs.to(device)
                    targets.to(device)

                    output, val_h = net.forward(inputs, val_h)

                    val_loss = criterion(output, targets.view(batch_size*n_steps))


                    val_losses.append(val_loss.item())

                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))
```


Initialize the model and run training for fixed number of epochs.


```python
# Your code here
```


```python
print(model)
```

    CharRNN(
      (lstm): LSTM(28, 256, num_layers=2, batch_first=True, dropout=0.5)
      (dropout): Dropout(p=0.5, inplace=False)
      (fc): Linear(in_features=256, out_features=28, bias=True)
    )


### **Task 5 - Generating Name - 5 points**

Now we can generate a dinosaur name using the trained model.


```python
import torch.nn.functional as F

def predict(char, model, id_to_token, token_to_id, h=None, device='cpu'):
    '''
    Predict the next character for the current character char
    (returns the predicted character and the hidden state).
    '''
    model.to(device)

    if h is None:
        h = model.init_hidden(1, device)

    x = np.array([[token_to_id[char]]])
    # your function for one-hot encoding
    inputs = one_hot_encode(x, len(token_to_id))

    inputs = torch.from_numpy(inputs)
    inputs.to(device)

    h = tuple([each.data for each in h])
    out, h = model.forward(inputs, h)

    p = F.softmax(out, dim=1).data

    p = p.cpu()

    top_ch = np.arange(len(token_to_id))

    p = p.numpy().squeeze()

    char = np.random.choice(top_ch, p=p/p.sum())

    return id_to_token[char], h
```


```python
def sample(model, size, id_to_token, token_to_id, prefix='<', device='cpu'):
    model.to(device)
    model.eval()

    chars = [ch for ch in prefix]

    h = model.init_hidden(1, device)

    for ch in prefix:
        char, h = predict(ch, model, id_to_token, token_to_id, h, device)

    chars.append(char)

    for i in range(size):
        char, h = predict(chars[-1], model, id_to_token, token_to_id, h, device)
        chars.append(char)
        if char == '>':
          break

    return ''.join(chars)
```

Try to generate names using different prefixes (also counts as the part of the task!).


```python
# Your code here
```

### **Task 6 - Sampling - 15 points**

Basic strategy for choosing next token after building probability distribution is **greedy search**. According to this approach we simply choose a token with the highest probability.
Here in `predict` function we applied **random sampling** to randomly choose the next token for prediction based on probability distribution obtained from softmax. Let's try out different generation strategies.

**Top-k sampling**

In random sampling we choose the next character using probability distribution over all characters of the vocabulary. We can limit the model to sample next token only among $k$ most probable tokens.

![top_k_sampling](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/top_k_sampling.png)



**Temperature**

We can try to affect the level of variation for our model generation.
Adding temperature parameter allows to smooth probability distribution so with higher temperature rare characters would appear more often.
This effect could be accomplished if we divide the logits by a temperature value.  Check [the link](https://lukesalamone.github.io/posts/what-is-temperature/) for more details.

**Task**

1. Modify functions predict and sample adding parameter top_k to consider  k  most probable characters. **Compare the results of generation using this approach to random sampling.**

2. Modify predict and sample functions for top-k sampling using parameter temperature.**Check the results of generation using sampling with different temperature values.**

Breifly write your insights.


```python
# Your code here
```



write your answers
