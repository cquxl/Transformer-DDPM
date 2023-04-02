import os
import argparse

from torch.backends import cudnn

from utils import *
from trainer import Trainer


def make_parser():
    '''
    '-':短参数，通常只是一个字符，可以合并-abc等价于-a -b -c，通常是参数的简写
    '--':长参数，是一个单词或一个短语，不能合并是可选参数，其他必须指定，通常是参数的全名
    like-->parser.add_argument('-f', '--file', dest='filename', help='Name of the file', required=True)
    命令行调用的时候可以使用-f或者--file,
    程序调用使用args.file
    '''
    parser = argparse.ArgumentParser('Transfomer-DDPM parser')
    # 输出的文件夹
    parser.add_argument("--output_dir", type=str, default='./TransformerDDPM_outputs1')  # 实验名字
    # 实验名字SN:Standard Norm, GC:Gaussian Copula, GC-t(Gaussian Copula+time_corr)
    parser.add_argument("-expn", "--experiment-name", type=str, default='PSM402-3-GC-B128-E150-noise-lr0.001-T1000-r-kmeans')  # 实验名字
    # 模型名字
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")  # 模型名字

    # 数据集
    parser.add_argument('--data_name', type=str, default='PSM')
    parser.add_argument('--data_path', type=str, default='./data/PSM')

    # 模型相关参数

    ## 扩散过程Standard Norm、Gaussian Copula
    parser.add_argument('--copula', type=str, default='Gaussian Copula', help='the copula method of diffusion process')
    parser.add_argument('--corr', type=str, default='feature corr', help='the corr kind of diffusion process')
    parser.add_argument('-T','--time_steps', type=int, default='1000', help='time steps of diffusion process')

    ## Diffusion Transformer
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate of Diffusion Transformer')
    parser.add_argument('-b', '--batch_size',type=int, default=128, help='batch size')

    # rmbda 扩散关联差异的系数
    parser.add_argument('--k', type=int, default=3)
    # 异常阈值ratio, kmeans为False，默认使用anomaly_ratio,否则使用kmeans自动计算
    parser.add_argument('--anomaly_ratio', type=int, default=1) # ratio需要调整，可以根据异常比例来调整
    parser.add_argument('--kmeans', default=True, action="store_true",help="calculate the anomaly ratio by kmeans")  # ratio需要调整，可以根据异常比例来调整
    # 计算异常分数是否采用逆转的x来计算mse，否则是用噪音的mse
    parser.add_argument('--reverse', default=False, action="store_true", help="caluculate mse of anamaly score by reverse x")

    # 有多少块gpu进行训练
    parser.add_argument('-d', '--device', type=int, default=0, help="device for training")

    # 是否接着训练，resume用于复原模型中途断开的重新训练
    parser.add_argument("--resume", default=False, action="store_true", help="resume training")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file") # 用于加载模型的pth文件
    parser.add_argument("-e", "--start_epoch", default=0, type=int, help="resume training start epoch")

    parser.add_argument('-epochs','--num_epochs', type=int, default=150)
    parser.add_argument('--win_size', type=int, default=100) # 时间窗口
    parser.add_argument('--slide_step', type=int, default=100)  # 滑动步长

    # 可保留，也可在程序中判断
    parser.add_argument('--input_c', type=int, default=25)
    parser.add_argument('--output_c', type=int, default=25)

    # 模型训练模式，可保留
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # 可通过实验名字进行保留，可保留
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    # 数据类型为float16默认为False-->则会使用float32
    parser.add_argument("--fp16", dest="fp16", default=False,action="store_true",
        help="Adopting mix precision training.",
    )

    return parser






def main():
    args = make_parser().parse_args()
    # args = vars(args)
    cudnn.benchmark = True # 提高计算运行效率
    # cudnn.deterministic = True # 避免结果波动
    if (not os.path.exists(args.output_dir)):
        mkdir(args.output_dir)
    if args.mode == 'train':
        trainer = Trainer(args)
        trainer.train()

if __name__ == '__main__':
    main()

