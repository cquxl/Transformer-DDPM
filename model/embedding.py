import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class TokenizedEmbedding(nn.Module):
    def __init__(self, f_in, d_model=512, local=True):
        super(TokenizedEmbedding, self).__init__()
        # circular表示循环填充，这样的填充法实际是再重复特征的某一个值，好处是保证特征权重分配均匀
        # 局部特征卷积其实也可以考虑1*1卷积,从某种角度上讲，我们并不知道特征附近的关系
        if local:
            self.token_conv = nn.Conv1d(in_channels=f_in, out_channels=d_model,
                                        kernel_size=3, padding=1, padding_mode='circular', bias=False)
        else:
            self.token_conv = nn.Conv1d(in_channels=f_in, out_channels=d_model,
                                        kernel_size=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d): # kaiming权重初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x): # x-->[64,100,25]，nn.Conv1d要求输入为[N,C,L)，因此需要颠倒位置
        token_x = self.token_conv(x.permute(0, 2, 1)) # [64,512,100]
        return token_x.transpose(1,2) # [64,100,512]

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1) # [5000,1]

        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)] # [1,100,512]


class Embedding(nn.Module):
    def __init__(self, f_in, d_model=512, dropout=0.0):
        super(Embedding, self).__init__()
        self.token_embedding = TokenizedEmbedding(f_in=f_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):  # [64,100,25]
        x = self.token_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
# 输入的维度是[batch_size,],有几张图
# 返回输出的维度是[batch_size, dim]
# 该模块被加入到每个残差模块
class SinusoidalPositionEmbeddings(nn.Module): # 与transfomer的embedding是有区别的
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)#维度256
        embeddings = time[:, None] * embeddings[None, :] # 维度扩充为[1,256]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # [1,dim]or[1,dim-1]
        return embeddings

if __name__ == '__main__':
    x = torch.randn(64, 100, 25)
    t = torch.randint(0,100,(64,1)) # [64,1,512]
    time_embed = SinusoidalPositionEmbeddings(512)
    embed_t = time_embed(t) # [64,512]
    print(embed_t.shape)

    # token_embed = TokenizedEmbedding(f_in=25, d_model=512, local=False)
    # token_x = token_embed(x)
    # print(token_x.shape)
    embed = Embedding(f_in=25, d_model=512, dropout=0.0)
    embed_x = embed(x)
    print(embed_x.shape)#[64,100,512]
    out_emb = embed_t+embed_x
    print(out_emb.shape)


