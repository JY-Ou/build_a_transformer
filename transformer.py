import os
import torch
from utils.ffn import *
from utils.attention import *
from utils.layernorm import *
from utils.positional import *
from utils.tokenized import *


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, pad, drop_prob, device):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, pad)
        self.pos_emb =  PositionalEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(drop_prob)

    def forward(self,x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)

        x = self.drop1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.drop2(x)
        x = self.norm2(x + _x)
        return x

# t_mask:下三角掩码，对未来信息的掩码，即因果掩码
# s_mask:对padding的掩码，padding的作用是统一句子的长度
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.attention1 = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_prob)

        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.drop3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        x = self.attention1(dec, dec, dec, t_mask)

        x = self.drop1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            _x = x
            x = self.cross_attention(x, enc, enc, s_mask)

            self.drop2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.drop3(x)
        x = self.norm3(x + _x)
        return x

class Encoder(nn.Module):
    def __init__(
            self, enc_voc_size, max_len, d_model,
            ffn_hidden, n_head, n_layer, drop_prob, device):
        super().__init__()
        self.embedding = TransformerEmbedding(enc_voc_size, d_model, max_len, 1, drop_prob, device)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, ffn_hidden, n_head, drop_prob)
                for _ in range(n_layer)
            ]
        )

    def forward(self, x, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x

class Decoder(nn.Module):
    def __init__(
            self, dec_voc_size, max_len, d_model,
            ffn_hidden, n_head, n_layer, drop_prob, device):
        super().__init__()
        self.embedding = TransformerEmbedding(dec_voc_size, d_model, max_len, 1, drop_prob, device)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, ffn_hidden, n_head, drop_prob)
                for _ in range(n_layer)
            ]
        )

        self.fc = nn.Linear(d_model, dec_voc_size)

    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)

        dec = self.fc(dec)
        return dec

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size,
                 max_len, d_model, n_heads, ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.encoder = Encoder(enc_voc_size, max_len, d_model, ffn_hidden, n_heads, n_layers, drop_prob, device)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden, n_heads, n_layers, drop_prob, device)

        self.src_pad_idx = src_pad_idx
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

    def forward(self, src, trg):
        # encoder mask
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        # decoder mask
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx
                                      ) * self.make_casual_mask(trg, trg)
        # cross attention mask
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)
        
        enc = self.encoder(src, src_mask)
        output = self.decoder(trg, enc, trg_mask, src_trg_mask)
        return output



