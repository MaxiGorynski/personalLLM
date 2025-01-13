GPT_CONFIG_124M = {
    "vocab size": 50257,        #Vocabulary size
    "context_length": 1024,     #Context length, how many tokens the model can handle via positional embeddings
    "emb_dim": 768,             #Embedding dimensions
    "n_heads": 12,              #Number of attention heads
    "n_layers": 12,             #Number of NN layers/transformer blocks
    "drop_rate": 0.1,           #Drouput rate of hidden units to prevent overfitting
    "qkv_bias": False           #Query-Key-Value bias
}