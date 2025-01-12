import re
from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

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

#Computing context vectors with trainable weights using x^2 as our initial query token]
x_2 = inputs[1] #Second input element
d_in = inputs.shape[1] #Input embedding size, d=3
d_out = 2 #The output embedding size, d_out = 2

#Initialise the three weight matrices, Wq for query, Wk for key, Wv for v value vectors
#Grad is set to false to reduce output clutter, but they should be set to True during model
#training so that matrices are updated
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
#print(query_2)

keys = inputs @ W_key
values = inputs @ W_value
#print("keys.shape:", keys.shape)
#print("values.shape:", values.shape)

keys_2 = keys[1] #Performing this on x^2 to start, so skipping to index 1, where x^2 is held
#Dot product of the query vector by the key vector
attn_scores_22 = query_2.dot(keys_2)
#print(attn_scores_22)

attn_scores_2 = query_2 @ keys.T
#print(attn_scores_2)

#Now, we turn attention scores to attention weights.
# We do this by scaling the attention scores
# (by dividing them by the square root of the embedding dimension of the keys)
# and using the softmax function
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
#print(attn_weights_2)

#Calculating the context vectors using the attention weights with matrix multiplication
context_vec_2 = attn_weights_2 @ values
#print(context_vec_2)

#Now, generalise the code to compute all context vectors in sequence

#...

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
#print(sa_v1(inputs))

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores/ keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
#print(sa_v2(inputs))

#Compute attention weights for masked attention
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores/keys.shape[-1] ** 0.5, dim=-1)
#print(attn_weights)

#Use tril to create a mask where values above diagonal are zero
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
#print(mask_simple)

#Multiply the mask with the attention weights to zero-out results above the diagonal
masked_simple = attn_weights*mask_simple
#print(masked_simple)

#Renormalise the attention weights to sum up to one
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple/row_sums
#print(masked_simple_norm)

#Make the thing more efficient by replacing zeros above the diagonal with negatinve infinit
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
#print(masked)

#Apply softmax to acquire the complete modified attention weights
attn_weights = torch.softmax(masked/keys.shape[-1] ** 0.5, dim=1)
#print(attn_weights)

#Apply dropout for attention weights to prevent overfitting
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.2)
example = torch.ones(6, 6)
#print(dropout(example))

#Apply dropout to the attention weight matrix
torch.manual_seed(123)
#print(dropout(attn_weights))

#Test CausalAttention is able to support multiple inputs
batch = torch.stack((inputs, inputs), dim=0)
#print(batch.shape)

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        #d_in = #Input feature dimensionality for each token (e.g., 512 for GPT models)
        self.d_out = d_out #Output dimensionality of the transformed token representations.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) #Current token focus
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias) #What current token is being matched againnst
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias) #What values the tokens communicate
        self.dropout = nn.Dropout(dropout) #Dropout layer added from SelfAttention_v1 #Randomly zeroinng several tokens
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)
        ) #Values above diag are blocked

    def forward(self, x):
        b, num_tokens, d_in = x.shape #Set up input tensor's batch size, token number, dimensionality. We transpose dimensions 1 and 2, keeping the batch dimennsion at the first position (0)

        # Each linear layer transforms the input tensor x into keys, queries, and values
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) #We transpose dimensions 1 and 2, keeping the batch dimennsion at the first position (0). Computes dot products between queries and keys for every pair of tokens in the sequence
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) #Prevent attention to future tokens by setting the corresponding attention scores to -inf.. Ops with a trailing underscore are performed in-place, avoiding unnecessary memory copies
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1) #Normalisation via converting scores into probabilities.
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values #Each token's context vector is computed as a weighted sum of the values, where the weights are given by the attention probabilities.
        return context_vec

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
#print("context_vecs.shape:", context_vecs.shape)

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention (
                d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)
context_length = batch.shape[1] #Number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
#print(context_vecs)
#print("context_vecs.shape:", context_vecs.shape)

class MultiHeadAttention(nn.Module):
    #Class constructor
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        #Assert to ensure that output dimension is divisible by number of heads
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        #Store output dimension, attenntion head number, dimensionality of each
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        #Initialise linear transformation layers of queries/keys/values
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        #Output projection after attention applied
        self.out_proj = nn.Linear(d_in, d_out)

        #Dropout layer for avoiding overfit
        self.dropout = nn.Dropout(dropout)

        #Mask to apply to scores for causal (self-attention) masking
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

        #Application of multi-head attention
        def forward(self, x):

            #Batch-size, number of tokens per sequences, embeddings size from input
            b, num_tokens, d_in = x.shape

            #Apply transformations to produce keys, queries, and values from x
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)

            #Reshape each of these for the tensor shape determined above
            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

            #Transpose to bring num_heads dimension to second axis for easier attention computation
            keys = keys.transpose(1, 2)
            queries = queries.transpose(1, 2)
            values = values.transpose(1, 2)

            #Compute attention scores through matrix mult of queries/keys
            attn_scores = queries @ keys.traspose(2, 3)

            #Create a boolean mask for the matrix above the diagonal
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

            #Fill masked positions with negative infinity, prevents future tokens being seen
            attn_scores.masked_fill_(mask_bool, -torch.inf)

            attn_weights = torch.softmax(
                attn_scores / keys.shape[-1]**0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            context_vec = (attn_weights @ values).transpose(1,2)

            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
            context_vec = self.out_prok(context_vec)
            return context_vec




