import torch
import numpy as np
import math
from torch.distributions.multivariate_normal import MultivariateNormal

def get_copula_noise(x_start, type='Standard Norm', mode='feature corr', cholesky=False):
    """"
    x_start --> [64,100,25]
    type --> copula kind
    mode --> 特征相关还是时间点相关
    注意要生成copula多元分布的噪音变量需要将协方差矩阵进行cholesky分解成l在于
    """
    b, time_point, feature_num = x_start.shape
    if type == "Standard Norm":
        noise = torch.rand_like(x_start)
        return noise
    elif type == 'Gaussian Copula':
        if mode == 'feature corr':
            # 边缘正态分布
            noise = []
            for i in range(b):
                e = torch.stack([torch.randn(time_point) for _ in range(feature_num)], dim=1) # 100 *25
                cov = e.T.cov() # [25,25]
                mvn = MultivariateNormal(loc=torch.zeros(feature_num), covariance_matrix=cov) # 不用多元分布来生成
                # 抽取多元分布的噪声
                copula_e = mvn.sample((time_point,)) # [100,25] # # 不用多元分布来生成
                noise.append(copula_e)
            noise = torch.stack(noise) # [64,100,25]
            return noise
        elif mode == 'time corr': # 采用径向基函数来模拟各时间点的相关性
            noise = []
            for i in range(b):
                cov = torch.zeros(time_point, time_point) # [100,100]
                for p in range(time_point):
                    for q in range(time_point):
                        cov[p,q] = rbk(p,q)
                mvn = MultivariateNormal(loc=torch.zeros(time_point), covariance_matrix=cov)
                copula_e = mvn.sample((feature_num, )).T
                noise.append(copula_e)
            noise = torch.stack(noise)  # [64,100,25]
            return noise
        else:
            raise ValueError("{} not supported.".format(mode))
    else:
        raise ValueError("copula version {} not supported.".format(type))

def cholesky(x_start, mean, cov):
    b, time_point, feature_num = x_start.shape
    # 对协方差矩阵进行 Cholesky 分解
    L = torch.linalg.cholesky(cov)
    # 生成标准正态分布的样本
    z = torch.randn(time_point, len(mean))
    x = mean + torch.matmul(z, L.T)
    return x



def rbk(t,u, gama=1):
    """
    径向基核
    return:时间节点i与时间节点j的相关性
    """
    return math.exp(-gama*(t-u)**2)

if __name__ == "__main__":
    x_start = torch.rand(64,100,25)
    noise = get_copula_noise(x_start, type='Standard Norm', mode='feature corr')
    print(noise.shape)

