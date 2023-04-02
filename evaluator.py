import torch
from tqdm import tqdm
import time
from collections import ChainMap, defaultdict
from utils import get_copula_noise
from model.ddpm import BetaSchedule, DDPM
import torch.nn.functional as F
from utils import my_kl_loss, s_p_loss, test_s_p_loss, my_noise_kl_loss
from utils import MeterBuffer, gpu_mem_usage, mem_usage
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.nn as nn
from data_load import get_data_loader, DataPrefetcher
from sklearn.cluster import KMeans
def time_synchronized():
    """pytorch-accurate time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

class Evaluator:
    """
    Transformer-DDPM Evaluation
    """

    def __init__(
            self,
            args,
            dataloader,
            # img_size: int,
            # confthre: float,
            # nmsthre: float,
            # num_classes: int,
            # testdev: bool = False,
            # per_class_AP: bool = True,
            # per_class_AR: bool = True,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            # img_size: image size after preprocess. images are resized
            #     to squares whose shape is (img_size, img_size).
            # confthre: confidence threshold ranging from 0 to 1, which
            #     is defined in the config file.
            # nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            # per_class_AP: Show per class AP during evalution or not. Default to True.
            # per_class_AR: Show per class AR during evalution or not. Default to True.
        """
        self.args = args
        self.dataloader = dataloader
        self.train_loader = get_data_loader(self.args.data_path, self.args.batch_size, win_size=self.args.win_size,
                                            slide_step=self.args.slide_step, mode='train', transform=True,
                                            dataset=self.args.data_name)
        self.thre_loader = get_data_loader(self.args.data_path, self.args.batch_size, win_size=self.args.win_size,
                                            slide_step=self.args.slide_step, mode='thre', transform=True,
                                            dataset=self.args.data_name)
        self.ddpm = DDPM(time_steps=args.time_steps, beta_schedule='sigmoid')
        self.print_interval = 10
        self.meter = MeterBuffer(window_size=self.print_interval) # 每隔10个iter刷新一次
        # self.img_size = img_size
        # self.confthre = confthre
        # self.nmsthre = nmsthre
        # self.num_classes = num_classes
        # self.testdev = testdev
        # self.per_class_AP = per_class_AP
        # self.per_class_AR = per_class_AR

    def evaluate(
        self, model, half=False
    ):
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        # ids = []
        # data_list = []
        output_data = {}
        loss_1 = []
        loss_2 = []
        epoch_nosie_loss = []

        progress_bar = tqdm

        inference_time = 0
        # nms_time = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for cur_iter, (input_data, labels) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # imgs = imgs.type(tensor_type)
                input = input_data.type(tensor_type)
                batch_size = input.shape[0]
                # t = torch.randint(0, self.args.time_steps, size=(batch_size // 2,))
                # t = torch.cat([t, self.args.time_steps - 1 - t], dim=0)  # t的形状（bz）
                t = torch.randint(0, self.args.time_steps,
                                  size=(batch_size // 1,))
                t = t.unsqueeze(-1).to(self.device)  # t的形状（bz,1）
                e = get_copula_noise(input, type=self.args.copula, mode=self.args.corr).to(self.device)
                x_t = self.ddpm.q_sample(input, t, noise=e).to(self.device)
                sqrt_alphas_cumprod = self.ddpm.sqrt_alphas_cumprod.to(self.device)
                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1 # true
                if is_time_record:
                    start = time.time()

                output, series, prior, _ = model(x_t, sqrt_alphas_cumprod, t)
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
                # 每个iter的推理时间
                loss_noise = F.mse_loss(e, output)
                # 用loss.py里的s_p_loss来计算
                series_loss, prior_loss = s_p_loss(series, prior, self.args.win_size)  # 返回的是均值
                loss1 = loss_noise - self.args.k * series_loss
                loss2 = loss_noise + self.args.k * prior_loss
                outputs = {}
                outputs['noise_loss'] = loss_noise
                outputs['loss1'] = loss1
                outputs['loss2'] = loss2

                epoch_nosie_loss.append(loss_noise.item())
                loss_1.append(loss1.item())
                loss_2.append(loss2.item())

                self.meter.update(
                    infer_time=infer_end - start,
                    inference_time=inference_time,
                    **outputs,
                )
        output_data['noise_loss'] = np.average(epoch_nosie_loss)
        output_data['loss1'] = np.average(loss_1)
        output_data['loss2'] = np.average(loss_2)
        return output_data
    def get_anomaly_score(self, model, loader, add_labels=False):
        tensor_type = torch.cuda.FloatTensor
        model = model.eval()
        temperature = 50 # 明显是trick参数
        criterion = nn.MSELoss(reduce=False)
        print("======================TEST MODE======================")
        # 统计训练集的数据
        attens_energy = []
        progress_bar = tqdm
        labels_list = []
        for cur_iter, (input_data, labels) in enumerate(
                progress_bar(loader)
        ):
            with torch.no_grad():
                input = input_data.type(tensor_type)
                batch_size = input.shape[0]
                # t = torch.randint(0, self.args.time_steps, size=(batch_size // 2,))
                # t = torch.cat([t, self.args.time_steps - 1 - t], dim=0)  # t的形状（bz）
                t = torch.randint(0, self.args.time_steps,
                                  size=(batch_size // 1,))
                t = t.unsqueeze(-1).to(self.device)  # t的形状（bz,1）
                e = get_copula_noise(input, type=self.args.copula, mode=self.args.corr).to(self.device)
                x_t = self.ddpm.q_sample(input, t, noise=e).to(self.device)
                sqrt_alphas_cumprod = self.ddpm.sqrt_alphas_cumprod.to(self.device)
                output, series, prior, _ = model(x_t, sqrt_alphas_cumprod, t)
                noise_loss = torch.mean(criterion(e, output), dim=-1) # [64,100]
                series_loss, prior_loss = test_s_p_loss(series, prior,self.args.win_size, temperature)
                metric = torch.softmax((-series_loss - prior_loss), dim=-1) # [64,100]
                # noise_kl_loss = my_noise_kl_loss(e, output) # [64,100]
                if self.args.reverse:
                    x_seq = self.p_sample_loop(model, input)
                    x_0 = x_seq[-1]
                    loss = torch.mean(criterion(x_0, input), dim=-1)
                    score = loss * metric
                score = noise_loss * metric  # [64,100]
                # score = torch.softmax(score, dim=-1)
                # score = loss*metric
                # score = torch.softmax(noise_kl_loss,dim=-1) * metric
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
                if add_labels:
                    labels_list.append(labels)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1) # 所有元素个数batch_size*len(loader)*100,(total,)
        attens_energy = np.array(attens_energy) # 实际与attens_energy维度是一致的(total,)
        if labels_list:
            labels_list = np.concatenate(labels_list, axis=0).reshape(-1)
            labels_list = np.array(labels_list)


        return attens_energy, labels_list
    def get_anomaly_ratio_by_kmeans(self, combined_energy):
        combined_energy = combined_energy.reshape(-1,1)
        class_pred = KMeans(n_clusters=2,random_state=42).fit_predict(combined_energy)
        # 计算anomaly_ratio
        anomaly_ratio = min(sum(class_pred) / len(class_pred), 1-sum(class_pred) / len(class_pred))
        return anomaly_ratio*100




    def get_thre(self, model):
        train_energy, _ = self.get_anomaly_score(model, self.train_loader, add_labels=False) # 不需要labels
        test_energy, test_labels = self.get_anomaly_score(model, self.thre_loader, add_labels=True)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        anomaly_ratio = self.get_anomaly_ratio_by_kmeans(combined_energy) # 比例乘了100
        print(f'anmaly_ratio:{anomaly_ratio}')
        # thre = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        thre = np.percentile(combined_energy, 100 - anomaly_ratio)
        # 寻找异常
        pred = (test_energy > thre).astype(int) # (total,)
        gt = test_labels.astype(int)
        return thre, pred, gt


    def detection_adjustment(self, model, anomaly_state=False): # trick
        """
        返回accuracy, precision, recall, f_score
        """
        thre, pred, gt = self.get_thre(model)
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True # 确定为异常
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
        # 计算Accuracy, prec, recall, f_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,average='binary')
        return thre, accuracy, precision, recall, f_score

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """
        :return:返回t-1时刻的生成样本
        """
        t = torch.tensor([t]).to(self.device)
        betas_t = self.ddpm.betas[t].to(self.device)
        sqrt_alphas_cumprod = self.ddpm.sqrt_alphas_cumprod.to(self.device)
        sqrt_one_minus_alphas_cumprod_t = self.ddpm.sqrt_one_minus_alphas_cumprod[t].to(self.device)
        sqrt_recip_alphas_t = self.ddpm.sqrt_recip_alphas[t].to(self.device)
        output, series, prior, _ = model(x_t, sqrt_alphas_cumprod, t)
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * output / sqrt_one_minus_alphas_cumprod_t).to(self.device)

        # z = torch.randn_like(x_t).to(device)
        z = get_copula_noise(x_t, type=self.args.copula, mode=self.args.corr).to(self.device)
        sigma_t = betas_t.sqrt()
        return model_mean + sigma_t * z

    @torch.no_grad()
    def p_sample_loop(self, model, input):
        # 从白噪声开始恢复样本
        # input_shape = input.shape
        x = get_copula_noise(input, type=self.args.copula, mode=self.args.corr).to(self.device)
        # 从x=e开始生成
        x_seq = [x]  # 每生成一个样本就加入进去
        for t in reversed(range(self.args.time_steps)):
            x = self.p_sample(model, x, t)
            # x_seq.append(x.cpu().numpy())
            x_seq.append(x)
        return x_seq


if __name__ == "__main__":
    thre_loader = get_data_loader('./data/SMD', 128, win_size=100,
                                            slide_step=100, mode='train', transform=True,
                                            dataset='SMD')
    # print(len(thre_loader))
    print(next(iter(thre_loader)))






















