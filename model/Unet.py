import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F

device = torch.device('cuda')
def exists(x):
    return x is not None # None-->False,else True

def default(val, d): # 该函数要么返回val，要么返回d()或d
    if exists(val): #若非none，则返回其值
        return val
    return d() if isfunction(d) else d

def num_to_groups(num, divisor):
    groups = num // divisor #返回商
    remainder = num % divisor # 返回余数
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module): # f(x) + x
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        # nn.init.constant_(self.fn.weight.data, 1)
        # nn.init.constant_(self.fn.bias.data, 0.0)
        # xavier_normal_
        # nn.init.xavier_normal_(self.fn.weight.data)
        # nn.init.constant_(self.fn.bias.data, 0.0)
        # # kaiming权重初始化
        # nn.init.kaiming_uniform_(self.fn.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None): # 上采样
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),# _,_,2*h，2*w
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),# 宽高不变,-->_,dim,2*h，2*w
    )

def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )
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

    class WeightStandardizedConv2d(nn.Conv2d):  # 标准化权重,其他默认
        """
        https://arxiv.org/abs/1903.10520
        weight standardization purportedly works synergistically with group normalization
        """

        def forward(self, x):
            eps = 1e-5 if x.dtype == torch.float32 else 1e-3

            weight = self.weight
            mean = reduce(weight, "o ... -> o 1 1 1", "mean")  # 通道，高，宽求均值，
            var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
            normalized_weight = (weight - mean) * (var + eps).rsqrt()

            return F.conv2d(
                x,
                normalized_weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)  # 标准权重化卷积
        x = self.norm(x)  # 通道分组归一化

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)  # 激活
        return x

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)  # 将channel维度切分成2块

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5  # 1/sqrt(32)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)  # 分成3组分别为q,k,v
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale  # [1,4,32,h*w]

        sim = einsum("b h d i, b h d j -> b h i j", q, k)  # 点积[1,4,h*w,h*w]
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()  # [1,4,h*w,h*w]
        attn = sim.softmax(dim=-1)  # [1,4,h*w,h*w]

        out = einsum("b h i j, b h d j -> b h i d", attn, v)  # [1,4,h*w,32]
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)  # [1,4*132, h, w]
        return self.to_out(out)  # [1,3,h,w]


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )  # [1,4,32,h*w]

        q = q.softmax(dim=-2)  # [1,4,32,h*w]
        k = k.softmax(dim=-1)  # [1,4,32,h*w]

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)  # [1,4,32,32]

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)  # [1,4,32, h*w]
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)  # #[1,4*32,h,w]
        return self.to_out(out)  # #[1,3,h,w]
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels # 输入的通道数
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim) #
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0) # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)] # [None, 3, 6, 12, 24]
        in_out = list(zip(dims[:-1], dims[1:])) # [(None, 3), (3, 6), (6, 12), (12, 24)]

        block_klass = partial(ResnetBlock, groups=resnet_block_groups) #分4组

        # time embeddings
        time_dim = dim * 4 # dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([]) # 4个
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1] # 24
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1) # 维度增加

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

if __name__ == '__main__':
    model = Unet(dim=32,
                 init_dim=None,
                 out_dim=None,
                 dim_mults=(1, 2, 4, 8),
                 channels=3,
                 self_condition=False,
                 resnet_block_groups=4, ).to(device)
    x = torch.rand(1, 3, 224, 224).to(device)
    time = torch.tensor([100])
    time = time.to(device)
    noise = model(x, time)
    print(noise.shape)