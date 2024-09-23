import torch
from torch import nn



# maxlen:模型在训练时将考虑的最大序列长度。超过这个长度的序列将被截断，而短于这个长度的序列将被填充（padding）到这个长度

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        # 初始化为0 shape:(max_len, d_model) -> 每行为一个词(token)，列为每个词的embedding
        self.encoding = torch.zeros(max_len, d_model).to(device)
        self.encoding.requires_grad_(False)

        # 生成(max_len,max_len)的矩阵
        pos = torch.arange(0, max_len).to(device)
        pos = pos.float().unsqueeze(1)
        # 生成0,2,4..d_model  d_model一般是偶数
        _2i = torch.arange(0, d_model, 2).to(device)

        # 按公式生成对应的位置编码
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model))).to(device)
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model))).to(device)

    def forward(self, x):
        # x.shape(b, num_token, d_model)
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]

