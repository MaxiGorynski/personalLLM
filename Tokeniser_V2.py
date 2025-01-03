import re
from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

with open ("the-verdict.txt.rtf", "r", encoding="utf-8") as f:
    raw_text = f.read()
#print ("Total number of characters:", len(raw_text))
#print (raw_text[:99])

#Simple tokenisation
text = "Hello world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
#print(result)

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
#print(len(preprocessed))
#print(preprocessed[:30])

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
#print(vocab_size)

vocab = {token:integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    #print(item)
    if i >= 50:
        break

import re


class SimpleTokeniserV1:
    def __init__(self, vocab):
        # Stores the vocab as a class attribute for access in the encode and decode methods
        self.str_to_int = vocab
        # Creates an inverse vocabulary that maps token IDs back to tokens
        self.int_to_str = {i: s for s,i in vocab.items()}

    # Processes input text into token ids
    def encode(self, text):
        # Updated regex for correct tokenization
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Removes spaces before specified punctuation
        text = re.sub(r'\s+([,.?!"()\'])',r'\1', text)
        return text


tokeniser = SimpleTokeniserV1(vocab)
text = """It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokeniser.encode(text)
#print(ids)
#print(tokeniser.decode(ids))

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
#print(len(vocab.items()))
for i, item in enumerate(list(vocab.items()) [-5:]):
    None
    #print(item)

class SimpleTokeniserV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # Replaces unknown words with unk tokens
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        #Replaces spaces before specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

text1 = "Hello, would you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
#print(text)

tokeniser = SimpleTokeniserV2(vocab)
#print(tokeniser.encode(text))
#print(tokeniser.decode(tokeniser.encode(text)))

#print("tiktoken version: ", version("tiktoken"))

tokeniser = tiktoken.get_encoding("gpt2")

text = ("Akwirw <|endoftext|> ier.")
integers = tokeniser.encode(text, allowed_special={"<|endoftext|>"})
#print(integers)
strings = tokeniser.decode(integers)
#print(strings)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokeniser.encode(raw_text)
#print(len(enc_text))
enc_sample = enc_text[50:]

context_size = 4 #Determines how many tokens are included in the input
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
#print(f"x: {x}")
#print(f"y:       {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    #print(context, "--->", desired)
    #print(tokeniser.decode(context)), "--->", tokeniser.decode([desired])

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokeniser, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        #Encodes entire text
        token_ids = tokeniser.encode(txt)

        #Uses a sliding window algo to chunk the book into overlapping sequences of max.length
        for i in range (0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    #Returns total number of rows in the dataset
    def __len__(self):
        return len(self.input_ids)

    #Returns a single row from the dataset
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    #Initialises the tokeniser
    tokeniser = tiktoken.get_encoding("gpt2")

    #Creates our dataset
    dataset = GPTDatasetV1(txt, tokeniser, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last, #Drops last batch if it is shorter than the specified batch_size to prevent loss spikes during training
        num_workers=num_workers #The number of CPU processes to use for preprocessing
    )

    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4,
                                  stride=1, shuffle=False)
data_iter = iter(dataloader) #Converts dataloader into Python iterator to fetch next entry via next() function
first_batch = next(data_iter)
second_batch = next(data_iter)
#print(first_batch)
#print(second_batch)

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4,
                                  stride=4, shuffle=False)
data_iter = iter(dataloader) #Converts dataloader into Python iterator to fetch next entry via next() function
inputs, targets = next(data_iter)
#print("Inputs:\n", inputs)
#print("Targets:\n", targets)

input_ids = torch.tensor([2,3,5,1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
#The weight matrix of the embedding layer shown through the following print statement
#shows a bunch of small random values. These will be adjusted an optimised through training.
#print(embedding_layer.weight)
#print(embedding_layer(torch.tensor(input_ids)))

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, stride=max_length, max_length=max_length, shuffle=False
)

#Embeds token IDs into a tensor with 8x4 dimensions
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
#print("Token IDs:\n", inputs)
#print("\nInputs shape: \n", inputs.shape)

#Embed these token IDs into 256-dimensional vectors
token_embeddings = token_embedding_layer(inputs)
#print(token_embeddings.shape)

#Create an additional layer with the same embedding dimension as the token_embedding_layer
#Input to pos_embeddings is usually a placeholder value, torch.arange etc etc
#Context_length is a variable representing supported input size of the LLM
#Input text can be longer than supported context length, which requires text truncation
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
#print(pos_embeddings.shape)

#Now, we add the positional embedding tensor of four 256-dim vecotrs to the token embeddings...
#...adding 4x256-dim pos_embeddings tensor to each 4X256-dim toke embedding tensor...
#...in each of the 8 batches

input_embeddings = token_embeddings + pos_embeddings
#print(input_embeddings.shape)

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]
#The second input token serves as the query
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
#print(attn_scores_2)

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
#print("Attention weights:", attn_weights_2)
#print("Sum:", attn_weights_2.sum())

query = inputs[1]
#The second input token serves as the query
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
#print(context_vec_2)

attn_scores = torch.empty(6, 6) #Empty tensor for storing vectors for all six words in the sentence relative to each other

#This double loop computes the dot product for every pair of tokens, treating one as the key token, one as the query token
#Uses for loops, which are a little slow
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
#print(attn_scores)

#Alternative using matrix multiplication
attn_scores = inputs @ inputs.T
#print(attn_scores)

#Now, we normalise the rows
#We set dims to -1 as a means of instructing the softmax function to apply normalisation along the last dimension of the attn_scores tensor.
#-1 means the last dimension of the 6x6 tensor, which is 'columns'. This means normalisation is done row-by-row...
#...across the columns. We want every row to sum to 1
attn_weights = torch.softmax(attn_scores, dim=-1)
#print(attn_weights)

#Verify that normalisation was successful by testing that all rows sum to 1
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
#print("Row 2 sum:", row_2_sum)
#print("All row sums:", attn_weights.sum(dim=-1))

#Finally, we use the attention weights to compute all context vectors via matrix multiplication
all_context_vecs = attn_weights @ inputs
#print(all_context_vecs)



