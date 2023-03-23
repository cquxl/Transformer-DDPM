import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os
# from ddpm import BetaSchedule
from .ddpm import BetaSchedule


class TriangularCausalMask():# 该类实现上三角不含对角线的mask
    def __init__(self, B, L, device="cuda"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class DiffusionAnomalyAttention(nn.Module): #返回的是还没有合并多头的注意力[64,100,8,64]
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(DiffusionAnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j) # self.distance---> [window_size, window_size] -->[100,100]对角阵

    # def forward(self, queries, keys, values, sigma, attn_mask, sqrt_alphas_cumprod)
    def forward(self, queries, keys, values, sigma, attn_mask, sqrt_alphas_cumprod, t): #
        B, L, H, E = queries.shape # [64,100,8, 64]
        _, S, _, D = values.shape # # [64,100,8,64]
        scale = self.scale or 1. / sqrt(E) # 1/8

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) # [64, 8, 100, 100]
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device) # [64,1,100,100]
            scores.masked_fill_(attn_mask.mask, -np.inf) # 将mask为1的位置在对应的scores上用-inf进行掩码
            # scores.masked_fill_(attn_mask.mask, 1e-9)
        attn = scale * scores # 掩码的位置也许是用来后面的计算的

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L[64,8,100]
        window_size = attn.shape[-1] # 100, 序列长度
        sigma = torch.sigmoid(sigma * 5) + 1e-5   # why?收缩到0到1
        sigma = torch.pow(3, sigma) - 1    # 3^sigma-->(0,2)
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L-->[64,8,100,100]

        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t]
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1).unsqueeze(-1).repeat(1, H, window_size, window_size)
        sigma_t = sqrt_alphas_cumprod_t * sigma

        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda() # [64,8,100,100]
        # prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma_t) * torch.exp(-prior ** 2 / 2 / (sigma_t ** 2))
        # 很明显prior在反向传播sigma的线性映射

        series = self.dropout(torch.softmax(attn, dim=-1)) # [64,8,100,100]
        V = torch.einsum("bhls,bshd->blhd", series, values)# [64,8,100,100],[64,100,8,64]-->[64,100,8,64]

        if self.output_attention:
            # return (V.contiguous(), series, prior, sigma)
            return (V.contiguous(), series, prior, sigma_t)
        else:
            return (V.contiguous(), None)

class DiffusionAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(DiffusionAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads) # 512/8 = 64
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        # 这里需要改正，他的意思是每个这个sigma也是原始X的变换对不对？
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    # def forward(self, queries, keys, values, attn_mask):
    def forward(self, queries, keys, values, attn_mask, sqrt_alphas_cumprod, t):
        B, L, _ = queries.shape # B=64, L=100, _=512
        _, S, _ = keys.shape #_=64, S=100, _=512
        H = self.n_heads # 8
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1) # [64,100,8,64]
        keys = self.key_projection(keys).view(B, S, H, -1) # [64,100,8,64]
        values = self.value_projection(values).view(B, S, H, -1)# # [64,100,8,64]
        sigma = self.sigma_projection(x).view(B, L, H) # # [64,100,8]

        # out, series, prior, sigma = self.inner_attention(
        #     queries,
        #     keys,
        #     values,
        #     sigma,
        #     attn_mask
        # ) # series, prior, sigma-->None，除非output_attention为True,训练阶段是True的
        # # out-->[64,100,8,64]
        out, series, prior, sigma_t = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask,
            sqrt_alphas_cumprod,
            t
        )

        out = out.view(B, L, -1) # [64,100,512]

        # out将多头concat后再进行线性映射，最后得到多头注意力self.out_projection(out)
        # 注意series是注意力机制的q*k,prior是基于距离的两者的输出都是[64,8,100,100],, sigma是[64,8,100,100]
        return self.out_projection(out), series, prior, sigma_t





if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    betas = BetaSchedule(time_steps=100).sigmoid_beta_schedule().to(device)
    # alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # [1000,]
    t = torch.randint(0, 100, size=(64 // 2,))
    t = torch.cat([t, 100 - 1 - t], dim=0)  # t的形状（bz）
    t = t.unsqueeze(-1)  # t的形状（bz,1）

    mask = TriangularCausalMask(B=64, L=100)
    print(mask.mask.shape)
    attention = DiffusionAnomalyAttention(win_size=100, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=True).to(device)
    # queries, keys, values, sigma, attn_mask
    q = torch.rand(64,100,8,64,device=device)
    k = torch.rand(64,100,8,64,device=device)
    v = torch.rand(64,100,8,64,device=device)
    sigma = torch.rand(64,100,8,device=device)
    attn_mask=mask
    attn, series, prior, sigma_t= attention(q, k ,v , sigma, attn_mask, sqrt_alphas_cumprod, t)
    print(attn.shape)
    print(series.shape)
    print(prior.shape)
    print(sigma_t.shape)
    x = torch.rand(64,100,512).to(device)
    attention_layer = DiffusionAttentionLayer(attention, d_model=512, n_heads=8, d_keys=None,d_values=None).to(device)
    out, series, prior, sigma_t  = attention_layer(x, x, x, attn_mask, sqrt_alphas_cumprod, t)
    print(out.shape) # [64, 100 ,512]


