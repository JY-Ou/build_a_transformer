import torch
from torch import nn
import torch.nn.functional as F
import math

# 自注意力
class SelfAttention(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)

    def forward(self,x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attn_scores = q @ k.T
        attn_weights = F.softmax( attn_scores / k.shape[-1]**0.5, dim=-1) # -1为最后一纬度
        output = attn_weights @ v
        return output

# 多头注意力
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__()
        assert (d_model % n_head ==0)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.w_combine = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # shape: b, num_tokens, d_model
        b, num_tokens, d_model = q.shape
        # shape: b, num_tokens, d_model
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)

        # shape: b, n_head, num_tokens, head_dim
        q = q.view(b, num_tokens, self.n_head, self.head_dim).permute(0, 2 ,1, 3)
        k = k.view(b, num_tokens, self.n_head, self.head_dim).permute(0, 2 ,1, 3)
        v = v.view(b, num_tokens, self.n_head, self.head_dim).permute(0, 2 ,1, 3)

        score = q @ k.transpose(2,3)/math.sqrt(k.shape[-1])
        if mask is not None:
            # mask = torch.tril(torch.ones(num_tokens, num_tokens, dtype=bool))
            score = score.masked_fill(mask == 0 , -1e9)
        score = F.softmax(score, dim=-1) @ v

        # .contiguous() 保证张量连续存储
        score = score.permute(0, 2, 1, 3).contiguous().view(b, num_tokens, d_model)

        output = self.w_combine(score)
        return output

# GQA
class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_groups):
        super(GroupQueryAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_groups = n_groups

        assert d_model % n_heads == 0
        # 每组处理多少head
        self.n_heads_groups = self.n_heads // self.n_groups
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, self.n_groups * self.head_dim)
        self.w_v = nn.Linear(d_model, self.n_groups * self.head_dim)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def expand(self, data):
        # shape: b, n_groups, num_tokens, head_dim
        batch, time = data.shape[0], data.shape[2]
        # shape: b, n_groups, n_heads_groups, num_tokens, head_dim
        data = data[:, :, None, :, :].expand(batch, self.n_groups, self.n_heads_groups, time,
                                             self.head_dim).contiguous()
        # shape: b, n_groups * n_heads_groups, num_tokens, head_dim = b, n_heads, num_tokens, head_dim
        return data.view(batch, self.n_groups * self.n_heads_groups, time, self.head_dim)

    def forward(self, q, k, v, mask=None):
        # shape: b, num_tokens, d_model
        q = self.w_q(q)
        # shape: b, num_tokens, n_groups * head_dim
        k = self.w_k(k)
        v = self.w_v(v)

        # shape: b, num_tokens, d_model
        batch = q.shape[0]
        # shape: b, n_groups * n_heads_groups, num_tokens, head_dim = b, n_heads, num_tokens, head_dim
        q = q.view(batch, -1, self.n_groups * self.n_heads_groups, self.head_dim).permute(0, 2, 1, 3)
        # shape: b, n_groups, num_tokens, head_dim
        k = k.view(batch, -1, self.n_groups, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch, -1, self.n_groups, self.head_dim).permute(0, 2, 1, 3)

        # 升维,将KV矩阵变回与Q同一维度 b, n_heads, num_tokens, head_dim
        k = self.expand(k)
        v = self.expand(v)
        # score.shape = b, n_heads, num_tokens, num_tokens
        score = q @ k.transpose(2, 3) / math.sqrt(self.head_dim)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        # score.shape = b, n_heads, num_tokens, head_dim
        score = self.softmax(score) @ v
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, -1, self.d_model)
        output = self.w_combine(score)
        return output



if __name__ == '__main__':
    # test
    torch.manual_seed(123)

    inputs = torch.tensor(
        [[0.43, 0.15, 0.89, 0.89],
         [0.55, 0.87, 0.66, 0.89],
         [0.57, 0.85, 0.64, 0.89],
         [0.22, 0.58, 0.33, 0.89],
         [0.77, 0.25, 0.10, 0.89],
         [0.05, 0.80, 0.55, 0.89]]
    )
    batch = torch.stack((inputs, inputs), dim=0)

    d_in = inputs.shape[1]
    sa_v2 = MultiHeadAttention(4, 2)
    print(sa_v2(batch,batch,batch,mask=True))

