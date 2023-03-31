import torch
import torch.nn as nn
import torch.nn.functional as F


#
# from attn import DiffusionAnomalyAttention, DiffusionAttentionLayer
# from embedding import Embedding, SinusoidalPositionEmbeddings
# from ddpm import BetaSchedule

from .attn import DiffusionAnomalyAttention, DiffusionAttentionLayer
from .embedding import Embedding, SinusoidalPositionEmbeddings
from .ddpm import BetaSchedule



class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, sqrt_alphas_cumprod, t, attn_mask=None): # x-->[64,100,512]
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            t=t
        )
        # new_x-->[64, 100, 512]
        # attn-->[64, 8, 100, 100]
        # mask-->[64, 8, 100, 100]
        # sigma-->[64, 8, 100, 100]
        x = x + self.dropout(new_x) # 残差连接
        y = self.norm1(x)   # 层归一化 --> [64, 100, 512]
        # 全连接+relu+全连接
        # 这是全连接+relu
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1)))) # [64,512,100]
        # 这是全连接
        # y = self.dropout(self.conv2(y)).transpose(-1,1) # [64, 100, 512]
        y = self.dropout(self.conv2(y).transpose(-1, 1)) #
        # 再次残差连接

        return self.norm2(x + y), attn, mask, sigma #[64, 100, 512], [64, 8, 100, 100], [64, 8, 100, 100],[64, 8, 100, 100]


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, sqrt_alphas_cumprod, t, attn_mask=None): # x-->[64,100,512]
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers: # 3个，由于输入输出的维度保持一致，因此可以循环进行
            x, series, prior, sigma_t = attn_layer(x, attn_mask=attn_mask, sqrt_alphas_cumprod=sqrt_alphas_cumprod, t=t)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma_t)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list # x-->[64,100,512], series_list-->[3]


class DiffusionTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(DiffusionTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.x_embedding = Embedding(enc_in, d_model, dropout) #
        self.t_embedding = SinusoidalPositionEmbeddings(d_model)  #


        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    DiffusionAttentionLayer(
                        DiffusionAnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True) # 直接是全连接并且未使用偏差

    def forward(self, x, sqrt_alphas_cumprod, t): # x-->[64,100,25], t-->[64,1]
        x_emb = self.x_embedding(x) # [64,100,512]
        t_emb = self.t_embedding(t) # [64,1,512]
        enc_out = x_emb+t_emb # [64,100,512]
        # enc_out = self.embedding(x) # [64,100,512]
        enc_out, series, prior, sigmas = self.encoder(enc_out, sqrt_alphas_cumprod, t)
        enc_out = self.projection(enc_out) # [64,100,25]

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]














if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    betas = BetaSchedule(time_steps=100).sigmoid_beta_schedule().to(device)
    # alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # [1000,]
    t = torch.randint(0, 100, size=(64 // 2,))
    t = torch.cat([t, 100 - 1 - t], dim=0)  # t的形状（bz）
    t = t.unsqueeze(-1).to(device)  # t的形状（bz,1）
    x = torch.rand(64, 100, 25).to(device)

    model = DiffusionTransformer(win_size=100, enc_in=25, c_out=25, d_model=512, n_heads=8, e_layers=1, d_ff=512,
                                 dropout=0.0, activation='gelu', output_attention=True).to(device)
    score,series, prior,_ = model(x,sqrt_alphas_cumprod,t)
    print(score.shape)
    # series_loss = 0.0
    # prior_loss = 0.0










