import torch
from torch import nn

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model, pad):
        super().__init__(vocab_size, d_model, padding_idx=pad)