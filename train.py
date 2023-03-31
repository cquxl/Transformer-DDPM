import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

from model.ddpm import BetaSchedule, DDPM
from model.TransformerDDPM import DiffusionTransformer
from data_load.data_loader import AnomalyDataLoader
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
betas = BetaSchedule(time_steps=100).sigmoid_beta_schedule().to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)  # 1/根号alpha_t,[1000,]
# 开根号（累乘alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # [1000,]
# 开根号（1-累乘alphas)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
# 计算后验分布的方差beta_t的后验估计-->(1-alphas_cumprod_prev)/(1-alphas_cumprod)*betas
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  # [1000,]
time_steps = 100
# batch_size=64
win_size = 100
num_epochs = 500
batch_size = 256
anomaly_loader = AnomalyDataLoader(win_size=100, slide_step=100, batch_size=batch_size, mode='train')
train_loader = anomaly_loader.PSMLoader()
anomaly_loader_val = AnomalyDataLoader(win_size=100, slide_step=100, batch_size=batch_size, mode='val')
vali_loader = anomaly_loader_val.PSMLoader()
anomaly_loader_test = AnomalyDataLoader(win_size=100, slide_step=100, batch_size=batch_size, mode='test')
test_loader = anomaly_loader_test.PSMLoader()
anomaly_loader_thre = AnomalyDataLoader(win_size=100, slide_step=100, batch_size=batch_size, mode='thre')
thre_loader = anomaly_loader_thre.PSMLoader()
anormly_ratio = 4.0

def q_sample(x_start, t, noise=None):
    """
    x_start，t都是张量
    该函数用来给x_start加噪音
    return：t时刻加噪后样本
    """
    if noise is None:
        noise = torch.randn_like(x_start).to(device)

    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].to(device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(device)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

@torch.no_grad()
def p_sample(model, x_t, t):
    """
    :return:返回t-1时刻的生成样本
    """
    t = torch.tensor([t]).to(device)
    betas_t = betas[t].to(device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(device)
    sqrt_recip_alphas_t = sqrt_recip_alphas[t].to(device)
    output, series, prior, _ = model(x_t, sqrt_alphas_cumprod, t)
    model_mean = sqrt_recip_alphas_t * (x_t - betas_t * output / sqrt_one_minus_alphas_cumprod_t).to(device)

    z = torch.randn_like(x_t).to(device)
    # posterior_variance_t = self.posterior_variance[t].to(device)
    sigma_t = betas_t.sqrt()
    return model_mean + sigma_t * z

@torch.no_grad()
def p_sample_loop(model, input):
    # 从白噪声开始恢复样本
    input_shape = input.shape
    x = torch.randn(input_shape).to(device)
    x_seq = [x] # 每生成一个样本就加入进去
    for t in reversed(range(time_steps)):
        x = p_sample(model, x, t)
        # x_seq.append(x.cpu().numpy())
        x_seq.append(x)
    return x_seq

def my_kl_loss(p, q): #kl散度损失
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001)) # p*ln(p/q
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))} # 学习率减半
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2




def train():
    print("======================TRAIN MODE======================")
    params_path = './exp/PSM'
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    time_now = time.time()
    train_steps = len(train_loader)
    model = DiffusionTransformer(win_size=100, enc_in=25, c_out=25, d_model=512, n_heads=8, e_layers=1, d_ff=512,
                                 dropout=0.0, activation='gelu', output_attention=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(num_epochs):
        iter_count = 0
        loss1_list = []
        loss2_list = []
        epoch_time = time.time()
        model.train()
        epoch_nosie_loss = []
        for i, (input_data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            iter_count += 1
            input = input_data.float().to(device)

            batch_size = input.shape[0]
            t = torch.randint(0, time_steps, size=(batch_size // 2,))
            t = torch.cat([t, time_steps - 1 - t], dim=0)  # t的形状（bz）
            t = t.unsqueeze(-1).to(device)  # t的形状（bz,1）
            e = torch.randn_like(input).to(device)
            # cov = torch.stack([x.transpose(0,1).cov() for x in input])
            # e = torch.einsum('bls,bsd->bld',e, cov)

            # sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].to(device)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(device)
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].to(device)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(device)

            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1).repeat(1,input.shape[1],input.shape[2])
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1).repeat(1,input.shape[1],input.shape[2])
            x_t = sqrt_alphas_cumprod_t * input + sqrt_one_minus_alphas_cumprod_t * e
            # predicted_e = model(x_t, t.squeeze(-1))
            output, series, prior, _ = model(x_t, sqrt_alphas_cumprod, t)  # 4个输出,output是predicted_e
            # 噪音损失
            loss_noise = F.mse_loss(e, output)
            series_loss = 0.0
            prior_loss = 0.0
            k=3
            for u in range(len(prior)):  # 长度是encoder的长度
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               win_size)).detach())) + torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       win_size)).detach(),
                               series[u])))
                prior_loss += (torch.mean(my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)),
                    series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(), (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            # rec_loss = self.criterion(output, input)  # 值损失
            rec_loss = loss_noise
            epoch_nosie_loss.append(rec_loss.item())
            # 取平均值

            # loss1_list.append((rec_loss - k * series_loss).item())
            loss1 = rec_loss - k * series_loss # 代码detach的是p，结果论文写的是detach的S
            loss2 = rec_loss + k * prior_loss
            loss1_list.append(loss1.item())
            loss2_list.append(loss2.item())
            # loss1.backward(retain_graph=True)
            # loss2.backward()
            # optimizer.step()
            if (i + 1) % 100 == 0:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((num_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()
        # 训练集
        sw.add_scalar('training_noise_loss', sum(epoch_nosie_loss) / len(epoch_nosie_loss), epoch)
        sw.add_scalar('training_loss_1', sum(loss1_list) / len(loss1_list), epoch)
        sw.add_scalar('training_loss_2', sum(loss2_list) / len(loss2_list), epoch)
        # 验证集
        val_noise_loss, vali_loss1, vali_loss2 = val(model)
        sw.add_scalar('val_noise_loss', val_noise_loss, epoch)
        sw.add_scalar('val_loss_1', vali_loss1, epoch)
        sw.add_scalar('val_loss_2', vali_loss2, epoch)
        # 测试集，画出F1
        accuracy, precision, recall, f_score = test(model)
        sw.add_scalar('test accuracy', accuracy, epoch)
        sw.add_scalar('test precision', precision, epoch)
        sw.add_scalar('test recall', recall, epoch)
        sw.add_scalar('test f_score', f_score, epoch)


        if (epoch+1) % 10 == 0:
            print(f'epoch:{epoch + 1},train_noise_loss:{sum(epoch_nosie_loss) / len(epoch_nosie_loss)}, train_loss_1:{sum(loss1_list) / len(loss1_list)}, '
                  f'train_loss_2:{sum(loss2_list) / len(loss2_list)}')
            # print(f'epoch:{epoch + 1} train_loss_1,{sum(loss1_list) / len(loss1_list)}')
            # print(f'epoch:{epoch + 1} train_loss_2,{sum(loss2_list) / len(loss2_list)}')

            print(f'epoch:{epoch + 1} val_noise_loss:{val_noise_loss}, val_loss_1,{vali_loss1}, val_loss_2,{vali_loss2}')
            # print(f'epoch:{epoch + 1} val_loss_1,{vali_loss1}')
            # print(f'epoch:{epoch + 1} val_loss_2,{vali_loss2}')

            # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                # train_loss = np.average(loss1_list)


    print(f"总共花费时间:{time.time()-time_now}")

def val(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss_1 = []
    loss_2 = []
    epoch_nosie_loss =[]
    for i, (input_data, _) in enumerate(vali_loader):
        input = input_data.float().to(device)
        batch_size = input.shape[0]
        t = torch.randint(0, time_steps, size=(batch_size // 2,))
        t = torch.cat([t, time_steps - 1 - t], dim=0)  # t的形状（bz）
        t = t.unsqueeze(-1).to(device)  # t的形状（bz,1）
        e = torch.randn_like(input).to(device)
        # cov = torch.stack([x.transpose(0,1).cov() for x in input])
        # e = torch.einsum('bls,bsd->bld',e, cov)

        # sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].to(device)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(device)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].to(device)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(device)

        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1).repeat(1, input.shape[1], input.shape[2])
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1).repeat(1, input.shape[1],
                                                                                               input.shape[2])
        x_t = sqrt_alphas_cumprod_t * input + sqrt_one_minus_alphas_cumprod_t * e
        output, series, prior, _ = model(x_t, sqrt_alphas_cumprod, t)
        loss_noise = F.mse_loss(e, output)
        series_loss = 0.0
        prior_loss = 0.0
        k = 3
        for u in range(len(prior)):  # 长度是encoder的长度
            series_loss += (torch.mean(my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                           win_size)).detach())) + torch.mean(
                my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   win_size)).detach(),
                           series[u])))
            prior_loss += (torch.mean(my_kl_loss(
                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                        win_size)),
                series[u].detach())) + torch.mean(
                my_kl_loss(series[u].detach(), (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               win_size)))))
        series_loss = series_loss / len(prior)
        prior_loss = prior_loss / len(prior)

        # rec_loss = self.criterion(output, input)  # 值损失
        rec_loss = loss_noise
        epoch_nosie_loss.append(rec_loss.item())
        # 取平均值

        # loss1_list.append((rec_loss - k * series_loss).item())
        loss1 = rec_loss - k * series_loss
        loss2 = rec_loss + k * prior_loss
        loss_1.append(loss1.item())
        loss_2.append(loss2.item())

        return np.average(epoch_nosie_loss), np.average(loss_1), np.average(loss_2)


def test(model):
    model.eval()
    temperature = 50
    print("======================TEST MODE======================")
    criterion = nn.MSELoss(reduce=False)
    attens_energy = []
    for i, (input_data, labels) in enumerate(test_loader):
        input = input_data.float().to(device)
        batch_size = input.shape[0]
        t = torch.randint(0, time_steps, size=(batch_size // 2,))
        t = torch.cat([t, time_steps - 1 - t], dim=0)  # t的形状（bz）
        t = t.unsqueeze(-1).to(device)  # t的形状（bz,1）
        e = torch.randn_like(input).to(device)
        # cov = torch.stack([x.transpose(0,1).cov() for x in input])
        # e = torch.einsum('bls,bsd->bld',e, cov)

        # sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].to(device)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(device)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].to(device)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(device)

        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1).repeat(1, input.shape[1], input.shape[2])
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1).repeat(1, input.shape[1],
                                                                                               input.shape[2])
        x_t = sqrt_alphas_cumprod_t * input + sqrt_one_minus_alphas_cumprod_t * e
        # predicted_e = model(x_t, t.squeeze(-1))
        output, series, prior, _ = model(x_t, sqrt_alphas_cumprod, t)  # 4个输
        loss_noise = F.mse_loss(e, output)
        loss_noise = torch.mean(loss_noise, dim=-1) # [64,100]
        # 注意你需要p_sample回去
        x_seq = p_sample_loop(model, input)
        x_0 = x_seq[-1]
        loss = torch.mean(criterion(input, x_0), dim=-1) # 计算原始mse损失 # [64,100]


        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                # [64,100] # 对头求平均
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               win_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               win_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)),
                    series[u].detach()) * temperature
        metric = torch.softmax((-series_loss - prior_loss), dim=-1) # [64,100]
        cri = metric * loss # 这里的loss应该改为nosie的所有损失，因为是拟合noise
        # cri = metric * loss_noise
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri) # anomaly_score
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = np.array(attens_energy)

    attens_energy = []
    for i, (input_data, labels) in enumerate(thre_loader):
        input = input_data.float().to(device)
        batch_size = input.shape[0]
        t = torch.randint(0, time_steps, size=(batch_size // 2,))
        t = torch.cat([t, time_steps - 1 - t], dim=0)  # t的形状（bz）
        t = t.unsqueeze(-1).to(device)  # t的形状（bz,1）
        e = torch.randn_like(input).to(device)
        # cov = torch.stack([x.transpose(0,1).cov() for x in input])
        # e = torch.einsum('bls,bsd->bld',e, cov)

        # sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].to(device)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(device)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].to(device)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(device)

        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1).repeat(1, input.shape[1], input.shape[2])
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1).repeat(1, input.shape[1],
                                                                                               input.shape[2])
        x_t = sqrt_alphas_cumprod_t * input + sqrt_one_minus_alphas_cumprod_t * e
        output, series, prior, _ = model(x_t, sqrt_alphas_cumprod, t)  # 4个输

        loss_noise = F.mse_loss(e, output)
        loss_noise = torch.mean(loss_noise, dim=-1) # [64,100]
        # 注意你需要p_sample回去
        x_seq = p_sample_loop(model, input)
        loss = torch.mean(criterion(input, x_seq[-1]), dim=-1) # 计算原始mse损失
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               win_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               win_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)),
                    series[u].detach()) * temperature
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - anormly_ratio) # 96分位数
        print("Threshold :", thresh)

        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(thre_loader):
            input = input_data.float().to(device)
            batch_size = input.shape[0]
            t = torch.randint(0, time_steps, size=(batch_size // 2,))
            t = torch.cat([t, time_steps - 1 - t], dim=0)  # t的形状（bz）
            t = t.unsqueeze(-1).to(device)  # t的形状（bz,1）
            e = torch.randn_like(input).to(device)
            # cov = torch.stack([x.transpose(0,1).cov() for x in input])
            # e = torch.einsum('bls,bsd->bld',e, cov)

            # sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].to(device)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(device)
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].to(device)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(device)

            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1).repeat(1, input.shape[1], input.shape[2])
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1).repeat(1, input.shape[1],
                                                                                                   input.shape[2])
            x_t = sqrt_alphas_cumprod_t * input + sqrt_one_minus_alphas_cumprod_t * e
            output, series, prior, _ = model(x_t, sqrt_alphas_cumprod, t)  # 4个输
            loss_noise = F.mse_loss(e, output)
            loss_noise = torch.mean(loss_noise, dim=-1)  # [64,100]
            # 注意你需要p_sample回去
            x_seq = p_sample_loop(model, input)
            loss = torch.mean(criterion(input, x_seq[-1]), dim=-1)  # 计算原始mse损失
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels) # 真实标签
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        pred = (test_energy > thresh).astype(int) # 大于他的为异常
        gt = test_labels.astype(int)
        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)
        # ?????
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1
        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))
        return accuracy, precision, recall, f_score























if __name__ == '__main__':
    train()

    # anomaly_loader = AnomalyDataLoader(win_size=100, slide_step=100, batch_size=512, mode='val')
    # val_loader = anomaly_loader.PSMLoader()
    # # print(len)










