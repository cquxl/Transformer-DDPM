import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from evaluator import Evaluator
from main import make_parser
from data_load import get_data_loader
from model.TransformerDDPM import DiffusionTransformer
from tqdm import tqdm
from utils import get_copula_noise
from model.ddpm import BetaSchedule, DDPM
from model.vaeLSTM import LSTMVAE, vae_loss
import torch.nn.functional as F
from utils import my_kl_loss, s_p_loss, test_s_p_loss, my_noise_kl_loss
from utils import MeterBuffer, gpu_mem_usage, mem_usage
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.nn as nn
from sklearn.cluster import KMeans
mean_f1 = {'Transformer-DDPM':96.00,'Anomaly-Transformer':94.79,
          'OmniAnomaly':85.16,'Beat-GAN':81.82, 'LSTM-VAE':56.72}
args = make_parser().parse_args()
train_loader = get_data_loader(args.data_path, args.batch_size, win_size=args.win_size,
                                            slide_step=args.slide_step, mode='train', transform=True,
                                            dataset=args.data_name)
test_loader = get_data_loader(args.data_path, args.batch_size, win_size=args.win_size,
                                            slide_step=args.slide_step, mode='test', transform=True,
                                            dataset=args.data_name)

evaluator = Evaluator(args, test_loader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffusionTransformer(win_size=args.win_size, enc_in=args.input_c,  c_out=args.output_c,
                             d_model=512, n_heads=8, e_layers=args.e_layers, d_ff=None, dropout=0.0,
                             activation='gelu', output_attention=True).to(device)
ddpm = DDPM(time_steps=args.time_steps, beta_schedule='sigmoid')


def detection_adjustment(model, anomaly_state=False, pth='MSL407-3-GC-B128-E150-noise-lr0.001-T1000-r-kmeans-diffTrue'):  # trick
    """
    返回accuracy, precision, recall, f_score
    """
    model_path = os.path.join(args.output_dir, pth,'last_epoch_ckpt.pth')
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    # model = model.load_state_dict(torch.load(model_path))
    model.eval()
    thre, anomaly_ratio, pred, gt = evaluator.get_thre(model)
    if args.detection:
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True  # 确定为异常
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
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    print(f'thre:{thre}, anomaly_ratio:{anomaly_ratio}')
    np.save(args.data_path+'/pred.npy', pred)
    return thre, anomaly_ratio, accuracy, precision, recall, f_score

def reconstruction(model, pth='MSL407-3-GC-B128-E150-noise-lr0.001-T1000-r-kmeans-diffTrue'):
    model_path = os.path.join(args.output_dir, pth,'last_epoch_ckpt.pth')
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    # model = model.load_state_dict(torch.load(model_path))
    model.eval()
    # thre, anomaly_ratio, pred, gt = evaluator.get_thre(model)
    x_0_list, noise_list, test_energy, test_labels = evaluator.get_anomaly_score(model, test_loader, add_labels=True)
    np.save(args.data_path + '/test_x_0_list_sn.npy', x_0_list)
    np.save(args.data_path+'/test_noise_list_sn.npy', noise_list)
    np.save(args.data_path + '/test_energy_sn.npy', test_energy)
    np.save(args.data_path + '/test_labels.npy', test_labels)
    return x_0_list, noise_list, test_energy


def plot_mean_f1(mean_f1):
    import d2l.torch as d2l
    d2l.use_svg_display()
    data = mean_f1
    bin_width = 0.5
    colors = ['#27296d', '#a393eb', '#f08a5d', '#f9ed69', '#66bfbf']
    index = 0
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    for key, value in data.items():
        # indexes_values = [round(value,5) for value in list(value.values())] # 获取各指标值
        # indexes_keys = list(value.keys())
        # xs = ind-(bin_width)*(1.5-index)
        plt.grid(True, linewidth=0.2)
        plt.bar(key, value, width=bin_width, label=key, color=colors[index])
        plt.text(key, value + 1, "%.2f" % value, ha='center', fontsize=10)  # plt.text 函数
        plt.ylim([55, 100])
        plt.ylabel('mean F1-Score')
        plt.xticks(rotation=15)
        # plt.xticks(fontsize=14)
        index += 1
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1);  ####设置上部坐标轴的粗细
    plt.savefig('D:/研究生生涯/学习/时间序列异常检测/Transformer-DDPM/fig/mean_f1.svg', dpi=300, bbox_inches='tight')
def plot_metrics(type='loss_1'):
    import d2l.torch as d2l
    d2l.use_svg_display()
    if type=='loss_1':
        colors = ['#112d4e', '#6a2c70', '#ffde7d', '#f67280']
        gc_diff_true_loss_1 = pd.read_csv('TransformerDDPM_final/MSL407-3-GC-B128-E150-noise-lr0.001-T1000-r1-diffTrue/loss_1.csv')
        gc_diff_false_loss_1 = pd.read_csv('TransformerDDPM_final/MSL407-3-GC-B128-E150-noise-lr0.001-T1000-r1-diffFalse/loss_1.csv')
        sn_diff_true_loss_1 = pd.read_csv('TransformerDDPM_final/MSL407-3-SN-B128-E150-noise-lr0.001-T1000-r1-diffTrue/loss_1.csv')
        sn_diff_false_loss_1 = pd.read_csv('TransformerDDPM_final/MSL407-3-SN-B128-E150-noise-lr0.001-T1000-r1-diffFalse/loss_1.csv')
        loss_list=[sn_diff_false_loss_1['Value'].values, gc_diff_false_loss_1['Value'].values,
                   sn_diff_true_loss_1['Value'].values, gc_diff_true_loss_1['Value'].values]
        label = ['Base', 'Gaussian Copula', 'DiffAssDis', 'Gaussian Copula & DiffAssDis']
        for i in range(4):
            plt.plot(loss_list[i], linewidth=1, color=colors[i], label=label[i])
            plt.grid(True, linewidth=0.2)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1);  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1);  ####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1);  ###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1);  ####设置上部坐标轴的粗细
        plt.legend(label)
        plt.show()
        plt.savefig('fig/MSL_loss1.svg', dpi=300, bbox_inches='tight')










if __name__ == "__main__":
    # test_x_0_list=np.load(r'D:\研究生生涯\学习\时间序列异常检测\Transformer-DDPM\data\MSL\test_x_0_list.npy').tolist()
    # print(test_x_0_list)
    # thre, anomaly_ratio, accuracy, precision, recall, f_score=detection_adjustment(model, anomaly_state=False, pth='MSL407-3-GC-B128-E150-noise-lr0.001-T1000-r-kmeans-diffTrue')
    # noise_list, test_energy, test_labels=reconstruction(model, test_loader, pth='SMD402-3-GC-B128-E150-noise-lr0.001-T1000-r0.5')
    # thre, anomaly_ratio, accuracy, precision, recall, f_score = detection_adjustment(model, anomaly_state=False,
    #                                                                                  pth='SMD402-3-GC-B128-E150-noise-lr0.001-T1000-r0.5')
    # plot_mean_f1(mean_f1)
    x_0_list, noise_list, test_energy = reconstruction(model,
                                                        pth='MSL407-3-SN-B128-E150-noise-lr0.001-T1000-r1-diffTrue')


