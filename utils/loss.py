import torch

def my_kl_loss(p, q): #kl散度损失
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001)) # p*ln(p/q
    return torch.mean(torch.sum(res, dim=-1), dim=1)
def my_noise_kl_loss(p,q): # p-->[64,100,512]
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))  # p*ln(p/q
    return torch.mean(res, dim=-1)

def s_p_loss(series, prior, win_size):
    series_loss = 0.0
    prior_loss = 0.0
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
    return  series_loss / len(prior), prior_loss / len(prior)

def test_s_p_loss(series, prior,win_size, temperature):
    series_loss = 0.0
    prior_loss = 0.0
    for u in range(len(prior)):
        series_loss += my_kl_loss(series[u], (
                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                       100)).detach()) * temperature
        prior_loss += my_kl_loss(
            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                    100)),
            series[u].detach()) * temperature
    return series_loss, prior_loss