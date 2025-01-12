#This MultiHeadAttention module allows us to use single matrix multiplications
#to compute keys, queries and values, instead of repeating it, as in a
#wrapper class. This saves on computationally expensive steps.


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
        self.out_proj = nn.Linear(d_out, d_out)

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
        attn_scores = queries @ keys.transpose(2, 3)

        #Create a boolean mask for the matrix above the diagonal
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        #Fill masked positions with negative infinity, prevents future tokens being seen
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        #Normalise attention scores, apply softmax to get weights
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)

        #Apply dropout for regularisationn
        attn_weights = self.dropout(attn_weights)

        #Compute weighted sum of values using attention weights/matrix mult.
        context_vec = (attn_weights @ values).transpose(1,2)

        #Rehsape context vector to original shape (batch_size, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        #Final output projection to obtain result of multi-head attention
        context_vec = self.out_proj(context_vec)
        return context_vec

torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)