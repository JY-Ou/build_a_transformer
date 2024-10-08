import matplotlib
import tiktoken
import torch
import torch.nn as nn
from transformer import Decoder

import os
import urllib.request

GPT_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "max_len": 256, # Context length
    "d_model": 768,         # Embedding dimension
    "ffn_hidden":4*768,     # ffn_hidden = 4*d_model
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_prob": 0.1,       # Dropout rate
    "pad": None,
    "trg_pad_idx": 1,
}


# decoder only
class GPTModel(nn.Module):
    def __init__(self, trg_pad_idx, dec_voc_size,
                 max_len, d_model, n_heads, ffn_hidden, n_layers, drop_prob, pad, device):
        super().__init__()
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden, n_heads, n_layers, drop_prob, pad, device)

        self.trg_pad_idx = trg_pad_idx
        self.device = device

    # 对padding的掩码
    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)

        # shape-> batch time len_q, len_k
        # ne:检查 q 中的每个元素是否不等于 pad_idx_q,不等于为true，等于为false
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)

        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)

        # 与运算，有padding的地方就取0
        mask = q & k
        return mask

    # torch,ones(len_q, len_k)生成 q*k大小的全1矩阵
    # torch.tril(torch,ones(len_q, len_k)) 生成q*k大小的下三角全为1
    def make_casual_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = (
            torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        )
        return mask

    def forward(self, trg):
        # decoder mask
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx
                                      ) * self.make_casual_mask(trg, trg)

        output = self.decoder(trg, None, trg_mask, None)
        return output


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

