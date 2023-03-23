"""
Author: Xiong Lang
Date: 2023-3-12
"""
# system having
import os
import random
# torch
import torch
from torch.utils.data import Dataset, DataLoader

# numpy, pandas
import numpy as np
import pandas as pd

# sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 导入异常检测数据集dataset类
# from dataset import AnomalyDataset
from .dataset import AnomalyDataset

# 设置种子
# source:https://blog.csdn.net/weixin_45928096/article/details/126938723

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed()


#-------------------------异常检测的DataLoader总类--------------------------
class AnomalyDataLoader:
    """
    该类用来加载各个数据集的DataLoader
    数据集：
    --1. PSM
    --2. SMAP
    --3. SMAP
    --4. SMD
    --5. ...
    """
    def __init__(self, win_size=100, slide_step=100, batch_size=64, mode='train'):
        # 数据集分割参数
        self.win_size = win_size
        self.slide_step = slide_step

        # dataloader设置
        self.batch_size = batch_size
        self.mode = mode

    def get_data_loader(self, dataset):
        if self.mode == 'train':
            shuffle = True
        else:
            shuffle = False
        data_loader = DataLoader(dataset,
                                 self.batch_size,
                                 shuffle,
                                 num_workers=0)
        return data_loader

    def MSLLoader(self):
        dataset = AnomalyDataset(mode=self.mode).MSLDataset()
        return self.get_data_loader(dataset)

    def PSMLoader(self):
        dataset = AnomalyDataset(mode=self.mode).PSMDataset()
        return self.get_data_loader(dataset)

    def SMAPLoader(self):
        dataset = AnomalyDataset(mode=self.mode).SMAPDataset()
        return self.get_data_loader(dataset)

    def SMDLoader(self):
        dataset = AnomalyDataset(mode=self.mode).SMDDataset()
        return self.get_data_loader(dataset)



if __name__ == '__main__':
    print('--------------MSL数据集------------------')
    msl_data_loader = AnomalyDataLoader().MSLLoader()

    print('--------------PSM数据集------------------')
    psm_data_loader = AnomalyDataLoader().PSMLoader()

    print('--------------SMAP数据集------------------')
    smap_data_loader = AnomalyDataLoader().SMAPLoader()

    print('--------------SMD数据集------------------')
    smd_data_loader = AnomalyDataLoader().SMDLoader()

    print("dataloader构造完毕")

