import torch
from torch import nn


# 平均值:mean 方差:var
# 平均值为 0，方差为 1  -> (x - mean) / math.sqrt(var)
# eps:避免在方差为0时出现除以零的错误
# 初始的scale（乘以1）和shift（加0）值没有任何影响,scale和shift是可训练的参数
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-10):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * out + self.shift
