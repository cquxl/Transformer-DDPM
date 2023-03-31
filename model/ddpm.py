import math
from inspect import isfunction
from functools import partial
import time

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.datasets import make_s_curve, make_swiss_roll
import os



os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
s_curve, _ = make_swiss_roll(10**4, noise=0.1)
s_curve = s_curve[:,[0,2]]
s_curve = (s_curve-s_curve.min(axis=0))/(s_curve.max(axis=0)-s_curve.min(axis=0))
dataset = torch.Tensor(s_curve).float()
batch_size=512
num_epoch=1000
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
lr = 1e-3
time_steps = 100
class BetaSchedule(object):
    """
    该类用来实现不同的beta取样计算,注意所有函数返回的都是一维张量
    """
    def __init__(self, time_steps):
        self.time_steps = time_steps

    def cosine_beta_schedule(self, s=0.08):
        """
        余弦调度
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """

        steps = self.time_steps + 1
        x = torch.linspace(0, self.time_steps, steps)  # [0,1,2,...,timesteps]
        alphas_cumprod = torch.cos(((x / self.time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def linear_beta_schedule(self):
        """
        线性调度
        """
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, self.time_steps)

    def quadratic_beta_schedule(self):
        """
        平方
        :return:
        """
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, self.time_steps) ** 2

    def sigmoid_beta_schedule(self):
        """
        sigmoid
        """
        beta_start = 0.00001
        beta_end = 0.005
        betas = torch.linspace(-6, 6, self.time_steps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


#自定义神经网络
class MLPDiffusion(nn.Module):
    def __init__(self, time_steps, num_features=2, num_units=128):
        super(MLPDiffusion, self).__init__()

        self.linears = nn.ModuleList(
            [
                # nn.Linear(2, num_units),
                nn.Linear(num_features, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                # nn.Linear(num_units, 2),
                nn.Linear(num_units, num_features)
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(time_steps, num_units),
                nn.Embedding(time_steps, num_units),
                nn.Embedding(time_steps, num_units),
            ]
        )

    def forward(self, x, t):
        #         x = x_0
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t) # [128,128],[128,128]，[128,128]
            x = self.linears[2 * idx](x) # [128,128]
            x += t_embedding # [128,128]，[128,128],[128,128] #很明显是一种位置编码
            x = self.linears[2 * idx + 1](x) # 激活
        # [128,128]
        x = self.linears[-1](x) # [128,2]

        return x # [128,2]

class EMA():
    """构建一个参数平滑器"""

    def __init__(self, mu=0.01):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

class DDPM:
    def __init__(self, time_steps=100, beta_schedule='sigmoid'):
        super(DDPM, self).__init__()
        # self.generator = model
        # self.input_shape = input_shape
        self.time_steps = time_steps
        self.beta_schedule = beta_schedule
        self.all_variable_dict = self.get_all_variable()

        # to_torch = partial(torch.tensor, dtype=torch.float32)
        # for variable, value in self.all_variable_dict.items():
        #     self.register_buffer(variable, value.float()) # register_buffer，无需梯度更新参数
        self.__dict__.update(self.all_variable_dict)



    def get_betas(self):
        if self.beta_schedule == 'cosine':
            betas = BetaSchedule(self.time_steps).cosine_beta_schedule()
        elif self.beta_schedule == 'linear':
            betas = BetaSchedule(self.time_steps).linear_beta_schedule()
        elif self.beta_schedule == 'quadratic':
            betas = BetaSchedule(self.time_steps).quadratic_beta_schedule()
        elif self.beta_schedule == 'sigmoid':
            betas = BetaSchedule(self.time_steps).sigmoid_beta_schedule()
        else:
            raise NotImplementedError(self.beta_schedule)
        return betas

    def get_all_variable(self):
        betas = self.get_betas()
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)#1/根号alpha_t,[1000,]
        # 开根号（累乘alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # [1000,]
        # 开根号（1-累乘alphas)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        # 计算后验分布的方差beta_t的后验估计-->(1-alphas_cumprod_prev)/(1-alphas_cumprod)*betas
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  # [1000,]
        return {'betas': betas, 'alphas': alphas,
                'alphas_cumprod': alphas_cumprod, 'alphas_cumprod_prev': alphas_cumprod_prev,
                'sqrt_recip_alphas': sqrt_recip_alphas, 'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
                'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod, 'posterior_variance': posterior_variance}

    # 前向过程
    def q_sample(self, x_start, t, noise=None):
        """
        x_start，t都是张量
        该函数用来给x_start加噪音
        return：t时刻加噪后样本
        """
        if noise is None:
            noise = torch.randn_like(x_start).to(device)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].to(device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(device)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1).repeat(1, x_start.shape[1], x_start.shape[2])
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1).repeat(1, x_start.shape[1],
                                                                                               x_start.shape[2])
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """
        :return:返回t-1时刻的生成样本
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        t = torch.tensor([t]).to(device)
        betas_t = self.betas[t].to(device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(device)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].to(device)
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * model(x_t, t) / sqrt_one_minus_alphas_cumprod_t).to(device)
        # 用copula生成

        z = torch.randn_like(x_t).to(device)
        # posterior_variance_t = self.posterior_variance[t].to(device)
        sigma_t = betas_t.sqrt()
        return model_mean + sigma_t * z
        # return model_mean + torch.sqrt(posterior_variance_t) * z


    @torch.no_grad()
    def p_sample_loop(self, x_start, model):
        # 从白噪声开始恢复样本
        # input_shape = x.shape
        x = torch.randn_like(x_start).to(device)
        x_seq = [x] # 每生成一个样本就加入进去
        for t in reversed(range(self.time_steps)):
            x = self.p_sample(model, x, t)
            x_seq.append(x.cpu().numpy())
        return x_seq

    def p_losses(self, model, x_start, loss_type="l2"):
        """
        对任一时刻t进行采样计算loss
        """
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.time_steps, size=(batch_size // 2,))
        t = torch.cat([t, self.time_steps - 1 - t], dim=0)
        t = t.unsqueeze(-1).to(device)  # t的形状（bz,1）
        # 获取对应时刻的值
        e  = torch.randn_like(x_start).to(device)
        # x_t = self.q_sample(x_start, t) #会再次生成噪声
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].to(device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(device)
        x_t= sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * e
        predicted_e = model(x_t, t.squeeze(-1))
        if loss_type == "l1":
            loss = F.l1_loss(e, predicted_e)
        elif loss_type == 'l2':
            loss = F.mse_loss(e, predicted_e)
            # loss = (e - predicted_e).square().mean()
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(e, predicted_e)
        else:
            raise NotImplementedError()

        return loss


def train():
    print('Training model...')
    model = MLPDiffusion(time_steps=100, num_features=2).to(device)
    ddpm = DDPM(input_shape=dataset.shape, time_steps=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    all_epoch_loss = []
    start = time.time()
    for epoch in range(num_epoch):
        epoch_loss = []
        for idx, batch_x in enumerate(dataloader):
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            loss = ddpm.p_losses(model, batch_x)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            epoch_loss.append(loss.item())
        # 计算单个epoch损失平均值
        epoch_loss_mean = sum(epoch_loss) / len(epoch_loss)
        all_epoch_loss.append(epoch_loss_mean)
        if (epoch % 200 == 0):
            print(epoch_loss_mean)
            x_seq = ddpm.p_sample_loop(model)
            fig, axs = plt.subplots(1, 10, figsize=(28, 3))
            for i in range(1, 11):  # [10,20,..100]
                x = x_seq[i * 10] # 已经是numpy
                axs[i - 1].scatter(x[:, 0], x[:, 1], color='red', edgecolor='white')
                axs[i - 1].set_axis_off()
                axs[i - 1].set_title('$q(\mathbf{x}_{' + str(i * 10) + '})$')
    plt.show()
    print(f"用时:{time.time() - start}s")


if __name__ == '__main__':
    # x_seq = train()
    # train()
    ddpm = DDPM(time_steps=100, beta_schedule='sigmoid')
    print(ddpm.sqrt_alphas_cumprod.shape)





































