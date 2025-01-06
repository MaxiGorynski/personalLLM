import re
from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

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