import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from loguru import logger
from tensorboardX import SummaryWriter
import shutil

from model.ddpm import BetaSchedule, DDPM
from model.TransformerDDPM import DiffusionTransformer
from model.vaeLSTM import LSTMVAE, vae_loss
from data_load import get_data_loader, DataPrefetcher
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import LRScheduler, setup_logger, get_world_size
from utils import get_local_rank, ModelEMA, get_rank, get_copula_noise
from utils import my_kl_loss, s_p_loss, test_s_p_loss
from utils import MeterBuffer, gpu_mem_usage, mem_usage
from utils import WandbLogger, log_metrics
from utils import save_checkpoint, adjust_status

from  evaluator import Evaluator

class Trainer:
    DEFAULTS = {}
    def __init__(self, args):
        '''

        :param args: 是parser解析器，相关参数在main.py，需要用这些参数建立模型并进行训练
        '''
        # 将args的参数进行传递
        # self.__dict__.update(Trainer.DEFAULTS, **args)
        self.args = args
        self.ddpm =DDPM(time_steps=args.time_steps, beta_schedule='sigmoid')
        self.device = 'cuda:{}'.format(args.device) # 'cuda:0'
        self.win_size = args.win_size # 100
        self.enc_in = args.input_c    # 25
        self.c_out = args.output_c    #25
        self.k = args.k  # 扩散关联差异系数lambda


        # exps
        self.warmup_epochs = 5 # 学习率预热
        self.warmup_lr = 0.
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.max_epoch = args.num_epochs
        self.distributed = get_world_size() > 1 # False因为只有一块gpu
        self.local_rank = get_local_rank() # 0
        self.rank = get_rank() # 0
        self.use_model_ema = True
        self.start_epoch = args.start_epoch # 默认是0
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.amp_training = args.fp16 # False
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.print_interval = 10 # 每个10个iter进行打印
        self.eval_interval = 1 # 每隔10个epoch进行验证
        self.meter = MeterBuffer(window_size=self.print_interval) # 每隔10个iter刷新一次
        # self.wandb_logger = WandbLogger.initialize_wandb_logger(self.args)

        self.scheduler = "warmcos"

        self.basic_lr_per_sequence = args.lr / args.batch_size # 1e-4/64

        self.input_size = (args.win_size, args.input_c) # [100, 25]
        self.file_name = os.path.join(args.output_dir, args.experiment_name) # ./TransformerDDPM_outputs/.._lr..batch_size..

        self.sw = SummaryWriter(logdir=self.file_name, flush_secs=5)
        self.best_loss = 0.0
        self.best_f1 = 0.0
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.anomaly_ratio = self.args.anomaly_ratio
        self.save_history_ckpt = False

        if (not os.path.exists(self.file_name)):
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=0,
            filename="train_log.txt",
            mode="a",
        )
    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best loss is {:.2f}".format(self.best_loss)
        )
        print(f"anmaly_ratio:{self.anomaly_ratio}, Accuracy:{self.accuracy}, Precision:{self.precision}, Recall:{self.recall}, best F1:{self.best_f1}")
    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def before_epoch(self):
        # 把loader给进行初始化
        self.train_loader = iter(self.train_loader)
        logger.info("---> start train epoch{}".format(self.epoch + 1))
        # 预热结束
        if self.epoch + 1 == self.max_epoch - self.warmup_epochs:
            logger.info("--->warm epoch end!")

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter() # pass
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()
        input_data, labels = self.prefetcher.next() # 在第二个epoch数据没有拿出来
        # float16半精度，内存占用更少，计算更快
        # 使用amp混合精度，提高算数精度
        input_data = input_data.to(self.data_type)
        labels= labels.to(self.data_type)

        input = input_data.to(self.device)
        batch_size = input.shape[0]
        # t = torch.randint(0, self.args.time_steps,
        #                   size=(batch_size // 2,))
        # t = torch.cat([t, self.args.time_steps - 1 - t], dim=0)  # t的形状（bz）
        t = torch.randint(0, self.args.time_steps,
                          size=(batch_size // 1,))
        t = t.unsqueeze(-1).to(self.device)  # t的形状（bz,1）

        # e = torch.randn_like(input).to(self.device)
        # 使用copula生成
        # e = get_copula_noise(input, type='Standard Norm', mode='feature corr').to(self.device)
        # e = get_copula_noise(input, type='Gaussian Copula', mode='feature corr').to(self.device)
        # e = get_copula_noise(input, type='Gaussian Copula', mode='time corr').to(self.device)
        e = get_copula_noise(input, type=self.args.copula, mode=self.args.corr).to(self.device)

        x_t = self.ddpm.q_sample(input, t, noise=e).to(self.device)

        sqrt_alphas_cumprod = self.ddpm.sqrt_alphas_cumprod.to(self.device)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training): # 混合精度训练
            if self.args.name == 'Transformer-DDPM':
                output, series, prior, _ = self.model(x_t, sqrt_alphas_cumprod, t)
            elif self.args.name == 'LSTM-VAE':
                x_decoded_mean, z_mean, z_log_sigma = self.model(input)
        if self.args.name == 'Transformer-DDPM':
            loss_noise = F.mse_loss(e, output)
            # 用loss.py里的s_p_loss来计算
            series_loss, prior_loss = s_p_loss(series, prior, self.win_size) # 返回的是均值

            loss1 = loss_noise - self.k * series_loss
            loss2 = loss_noise + self.k * prior_loss
            outputs={}
            outputs['noise_loss'] = loss_noise
            outputs['loss1'] = loss1
            outputs['loss2'] = loss2
            self.best_loss = loss_noise
            self.optimizer.zero_grad()
            self.scaler.scale(loss1).backward(retain_graph=True)
            self.scaler.scale(loss2).backward()
            self.scaler.step(self.optimizer)
        elif self.args.name == 'LSTM-VAE':
            rec_loss = vae_loss(input, x_decoded_mean, z_mean, z_log_sigma)
            outputs = {}
            outputs['rec_loss'] = rec_loss
            self.best_loss = rec_loss
            self.optimizer.zero_grad()
            self.scaler.scale(rec_loss).backward()
            self.scaler.step(self.optimizer)
        if self.use_model_ema:
            self.ema_model.update(self.model)
        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups: # 调整学习率
            param_group["lr"] = lr
        iter_end_time = time.time()
        # 每隔20个iter刷新一下
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def after_iter(self):
        if (self.iter + 1) % self.print_interval == 0:
            # 打印log信息
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))
            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )
            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )
            mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())
            logger.info(
                "{}, {}, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    mem_str,
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            if self.rank == 0:
                metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}
                metrics.update({
                    "train/lr": self.meter["lr"].latest
                })
                log_metrics(self.sw, metrics, self.epoch+1)
            self.meter.clear_meters()

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")
        if (self.epoch+1) % self.eval_interval == 0: # evaluate
            # self.evaluator.evaluate(model)
            self.evaluate_and_save_model()



    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
        with adjust_status(evalmodel, training=False):
            outputs_data = self.evaluator.evaluate(evalmodel)
            thre, anomaly_ratio, accuracy, precision, recall, f_score = self.evaluator.detection_adjustment(evalmodel, anomaly_state=False)
        if self.args.name == 'Transformer-DDPM':
            update_best_ckpt = outputs_data['noise_loss'] < self.best_loss
        elif self.args.name == 'LSTM-VAE':
            update_best_ckpt = outputs_data['rec_loss'] < self.best_loss
        update_best_f1 = f_score > self.best_f1
        if update_best_f1:
            self.accuracy = accuracy
            self.precision = precision
            self.recall = recall
            self.anomaly_ratio = anomaly_ratio
        if self.args.name == 'Transformer-DDPM':
            self.best_loss = min(outputs_data['noise_loss'], self.best_loss)
        elif self.args.name == 'LSTM-VAE':
            self.best_loss = min(outputs_data['rec_loss'], self.best_loss)
        self.best_f1 = max(self.best_f1, f_score)
        if self.rank == 0:
            if self.args.name == 'Transformer-DDPM':
                self.sw.add_scalar('val/loss1', outputs_data['loss1'], self.epoch+1)
                self.sw.add_scalar('val/loss2', outputs_data['loss2'], self.epoch + 1)
                self.sw.add_scalar('val/noise_loss', outputs_data['noise_loss'], self.epoch + 1)
            elif self.args.name == 'LSTM-VAE':
                self.sw.add_scalar('val/rec_loss', outputs_data['rec_loss'], self.epoch+1)
            self.sw.add_scalar('test/thre', thre, self.epoch + 1)
            self.sw.add_scalar('test/accuracy', accuracy, self.epoch + 1)
            self.sw.add_scalar('test/precision', precision, self.epoch + 1)
            self.sw.add_scalar('test/recall', recall, self.epoch + 1)
            self.sw.add_scalar('test/f_score', f_score, self.epoch + 1)
            # logger.info("\n" + summary)
        self.save_ckpt("last_epoch", update_best_ckpt)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}")



    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            }
            save_checkpoint(ckpt_state, update_best_ckpt, self.file_name, ckpt_name)

    def before_iter(self):
        pass

    def get_model(self):
        if self.args.name == "Transformer-DDPM":
            self.model = DiffusionTransformer(win_size=self.args.win_size, enc_in=self.enc_in,  c_out=self.c_out,
                                              d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0,
                                              activation='gelu', output_attention=True).to(self.device)
            return self.model
        if self.args.name == "LSTM-VAE":
            self.model = LSTMVAE(input_dim=self.enc_in, win_size=self.args.win_size, timesteps=self.args.time_steps,
                                 batch_size=self.args.batch_size, intermediate_dim=32, latent_dim=100, epsilon_std=1.0).to(self.device)
            return self.model

    def get_optimizer(self):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr # 0
            else:
                lr = self.basic_lr_per_sequence * self.args.batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.LayerNorm) or "ln" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay
            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def before_train(self): # 训练前需要记录logger的详细信息
        '''
        该函数只是在训练前设置好相关参数与模型和log配置，并没有进行训练
        '''
        logger.info("args: {}".format(self.args))
        model = self.get_model()
        if self.args.name == "Transformer-DDPM":
            self.optimizer = self.get_optimizer()
        elif self.args.name == "LSTM-VAE":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        # 注意resume过程必须要填写后才使用，否则从头训练
        if self.args.resume:
            model = self.resume_train(model) # 更新模型参数
        # 获取train_loader
        self.train_loader = get_data_loader(self.args.data_path, self.args.batch_size, win_size=self.win_size,
                                            slide_step=self.args.slide_step, mode='train', transform=True,
                                            dataset=self.args.data_name)
        self.val_loader = get_data_loader(self.args.data_path, self.args.batch_size, win_size=self.win_size,
                                            slide_step=self.args.slide_step, mode='val', transform=True,
                                            dataset=self.args.data_name)
        self.thre_loader = get_data_loader(self.args.data_path, self.args.batch_size, win_size=self.win_size,
                                            slide_step=self.args.slide_step, mode='thre', transform=True,
                                            dataset=self.args.data_name)
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        self.max_iter = len(self.train_loader)
        self.lr_scheduler = self.get_lr_scheduler(self.basic_lr_per_sequence * self.args.batch_size,
                                                  self.max_iter)
        # 是否多gpu并行计算
        if self.distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema: # 是否使用指数滑动平均， EMA本质是对变量的一种加权平均
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.evaluator = Evaluator(self.args, self.val_loader)

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def get_lr_scheduler(self, lr, iters_per_epoch):
        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr
        )
        return scheduler

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        """
        该函数用于由于训练过程中途断开时重新训练
        """
        # if self.args.resume:
        logger.info("resume training")
        if self.args.ckpt is None:
            ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
        else:
            ckpt_file = self.args.ckpt
        ckpt = torch.load(ckpt_file, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = (self.args.start_epoch - 1 if self.args.start_epoch is not None else ckpt["start_epoch"])
        self.start_epoch = start_epoch
        logger.info(
            "loaded checkpoint '{}' (epoch {})".format(
                self.args.resume, self.start_epoch
            )
        )
        return model

if __name__ == "__main__":
    # train_loader = get_data_loader(data_path='./data/PSM', batch_size=64, win_size=100, slide_step=100, mode='train', transform=True, dataset='PSM')
    # print(next(iter(train_loader))[-1].shape)
    x =torch.rand(64,100,25)
    lstm_vae = LSTMVAE(input_dim=25, win_size=100, timesteps=1000, batch_size=64, intermediate_dim=32, latent_dim=100, epsilon_std=1.0)
    x_decoded_mean, z_mean, z_log_sigma = lstm_vae(x)
    print(x_decoded_mean.shape)










