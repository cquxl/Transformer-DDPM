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

# 标准化
def scaler(default='StandardScaler'):
    if default == "MinMaxScaler":#默认为0到1
        return MinMaxScaler()
    return StandardScaler()

#-------------------------MSL,PSM,SMAP,SMD的类--------------------------
# 每个类的参数为：data_path, win_size, slide_step, mode='train', transform=True

class MSLSeqDataset(Dataset):
    """
    1. 该类实现异常检测MSL数据集的Dataset构造，训练集和测试集
    2. 数据集需要将时间序列定义好时间窗口以及滑动步数
    """

    def __init__(self, data_path, win_size, slide_step, mode='train', transform=True):
        """
        :data_path:数据集的路径-->like "./dataset/MSL"-->str
        :win_size:时间序列的窗口-->int
        :slide_step:时间序列滑动的步数-->int
        """
        # self.data_path = data_path
        self.win_size = win_size
        self.slide_step = slide_step
        self.mode = mode
        # self.scaler = StandardScaler() # 用于numpy的数据集
        self.scaler = scaler()

        # train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
        # test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
        # test_label_df = pd.read_csv(os.path.join(data_path, 'test_label.csv'))

        # 获取数据集的特征df的1：列, numpy格式
        # train_data = train_df.values[:, 1:]
        train_data = np.load(data_path + "/MSL_train.npy")
        # test_data = test_df.values[:, 1:]
        test_data = np.load(data_path + "/MSL_test.npy")
        # test_label_data = test_label_df.values[:, 1:]
        test_label_data = np.load(data_path + "/MSL_test_label.npy")

        # 处理缺失数据集nan
        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)
        test_label_data = np.nan_to_num(test_label_data)

        # 标准化数据集
        if transform:
            self.train = self.scaler.fit_transform(train_data)
            self.val = self.scaler.transform(test_data)
        else:
            self.train = train_data
            self.val = test_data

        self.test_labels = test_label_data

        print(f"train shape：{self.train.shape}")
        print(f"val shape：{self.val.shape}")
        print(f"test label shape：{self.test_labels.shape}")

    def __len__(self):
        """
        目标数据集的样本个数:注意会考虑时间序列窗口和滑动后的分割数据集个数

        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.slide_step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.slide_step + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.slide_step + 1

    def __getitem__(self, index):
        # 注意到有滑动步数，那么其实每一个序列都是从原始序列的index*slide_step的索引开始走的
        org_index = index * self.slide_step
        if self.mode == "train":
            return np.float32(self.train[org_index:org_index + self.win_size]), \
                np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[org_index:org_index + self.win_size]), \
                np.float32(self.test_labels[org_index:org_index + self.win_size])
        else:
            return np.float32(self.val[org_index:org_index + self.win_size]), \
                np.float32(self.test_labels[org_index:org_index + self.win_size])


class PSMSeqDataset(Dataset):
    """
    1. 该类实现异常检测PSM数据集的Dataset构造，训练集和测试集
    2. 数据集需要将时间序列定义好时间窗口以及滑动步数
    """

    def __init__(self, data_path, win_size, slide_step, mode='train', transform=True):
        """
        :data_path:数据集的路径-->like "./dataset/PSM"-->str
        :win_size:时间序列的窗口-->int
        :slide_step:时间序列滑动的步数-->int
        """
        # self.data_path = data_path
        self.win_size = win_size
        self.slide_step = slide_step
        self.mode = mode
        # self.scaler = StandardScaler() # 用于numpy的数据集
        self.scaler = scaler()

        train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
        test_label_df = pd.read_csv(os.path.join(data_path, 'test_label.csv'))

        # 获取数据集的特征df的1：列, numpy格式
        train_data = train_df.values[:, 1:]
        test_data = test_df.values[:, 1:]
        test_label_data = test_label_df.values[:, 1:]

        # 处理缺失数据集nan
        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)
        test_label_data = np.nan_to_num(test_label_data)

        # 标准化数据集
        if transform:
            self.train = self.scaler.fit_transform(train_data)
            self.val = self.scaler.transform(test_data)
        else:
            self.train = train_data
            self.val = test_data

        self.test_labels = test_label_data

        print(f"train shape：{self.train.shape}")
        print(f"val shape：{self.val.shape}")
        print(f"test label shape：{self.val.shape}")

    def __len__(self):
        """
        目标数据集的样本个数:注意会考虑时间序列窗口和滑动后的分割数据集个数

        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.slide_step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.slide_step + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.slide_step + 1

    def __getitem__(self, index):
        # 注意到有滑动步数，那么其实每一个序列都是从原始序列的index*slide_step的索引开始走的
        org_index = index * self.slide_step
        if self.mode == "train":
            return np.float32(self.train[org_index:org_index + self.win_size]), \
                np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[org_index:org_index + self.win_size]), \
                np.float32(self.test_labels[org_index:org_index + self.win_size])
        else:
            return np.float32(self.val[org_index:org_index + self.win_size]), \
                np.float32(self.test_labels[org_index:org_index + self.win_size])


class SMAPSeqDataset(Dataset):
    """
    1. 该类实现异常检测MSL数据集的Dataset构造，训练集和测试集
    2. 数据集需要将时间序列定义好时间窗口以及滑动步数
    """

    def __init__(self, data_path, win_size, slide_step, mode='train', transform=True):
        """
        :data_path:数据集的路径-->like "./dataset/SMAP"-->str
        :win_size:时间序列的窗口-->int
        :slide_step:时间序列滑动的步数-->int
        """
        # self.data_path = data_path
        self.win_size = win_size
        self.slide_step = slide_step
        self.mode = mode
        # self.scaler = StandardScaler() # 用于numpy的数据集
        self.scaler = scaler()

        # train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
        # test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
        # test_label_df = pd.read_csv(os.path.join(data_path, 'test_label.csv'))

        # 获取数据集的特征df的1：列, numpy格式
        # train_data = train_df.values[:, 1:]
        train_data = np.load(data_path + "/SMAP_train.npy")
        # test_data = test_df.values[:, 1:]
        test_data = np.load(data_path + "/SMAP_test.npy")
        # test_label_data = test_label_df.values[:, 1:]
        test_label_data = np.load(data_path + "/SMAP_test_label.npy")

        # 处理缺失数据集nan
        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)
        test_label_data = np.nan_to_num(test_label_data)

        # 标准化数据集
        if transform:
            self.train = self.scaler.fit_transform(train_data)
            self.val = self.scaler.transform(test_data)
        else:
            self.train = train_data
            self.val = test_data

        self.test_labels = test_label_data

        print(f"train shape：{self.train.shape}")
        print(f"val shape：{self.val.shape}")
        print(f"test label shape：{self.test_labels.shape}")

    def __len__(self):
        """
        目标数据集的样本个数:注意会考虑时间序列窗口和滑动后的分割数据集个数

        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.slide_step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.slide_step + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.slide_step + 1

    def __getitem__(self, index):
        # 注意到有滑动步数，那么其实每一个序列都是从原始序列的index*slide_step的索引开始走的
        org_index = index * self.slide_step
        if self.mode == "train":
            return np.float32(self.train[org_index:org_index + self.win_size]), \
                np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[org_index:org_index + self.win_size]), \
                np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.val[org_index:org_index + self.win_size]), \
                np.float32(self.test_labels[org_index:org_index + self.win_size])
        else:
            np.float32(self.val[
                       index // self.slide_step * self.win_size:index // self.slide_step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.slide_step * self.win_size:index // self.slide_step * self.win_size + self.win_size])



class SMDSeqDataset(Dataset):
    """
    1. 该类实现异常检测MSL数据集的Dataset构造，训练集和测试集
    2. 数据集需要将时间序列定义好时间窗口以及滑动步数
    """

    def __init__(self, data_path, win_size, slide_step, mode='train', transform=True):
        """
        :data_path:数据集的路径-->like "./dataset/SMD"-->str
        :win_size:时间序列的窗口-->int
        :slide_step:时间序列滑动的步数-->int
        """
        # self.data_path = data_path
        self.win_size = win_size
        self.slide_step = slide_step
        self.mode = mode
        # self.scaler = StandardScaler() # 用于numpy的数据集
        self.scaler = scaler()

        # train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
        # test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
        # test_label_df = pd.read_csv(os.path.join(data_path, 'test_label.csv'))

        # 获取数据集的特征df的1：列, numpy格式
        # train_data = train_df.values[:, 1:]
        train_data = np.load(data_path + "/SMD_train.npy")
        # test_data = test_df.values[:, 1:]
        test_data = np.load(data_path + "/SMD_test.npy")
        # test_label_data = test_label_df.values[:, 1:]
        test_label_data = np.load(data_path + "/SMD_test_label.npy")

        # 处理缺失数据集nan
        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)
        test_label_data = np.nan_to_num(test_label_data)

        # 标准化数据集
        if transform:
            self.train = self.scaler.fit_transform(train_data)
            self.val = self.scaler.transform(test_data)
        else:
            self.train = train_data
            self.val = test_data

        self.test_labels = test_label_data

        print(f"train shape：{self.train.shape}")
        print(f"val shape：{self.val.shape}")
        print(f"test label shape：{self.test_labels.shape}")

    def __len__(self):
        """
        目标数据集的样本个数:注意会考虑时间序列窗口和滑动后的分割数据集个数

        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.slide_step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.slide_step + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.slide_step + 1

    def __getitem__(self, index):
        # 注意到有滑动步数，那么其实每一个序列都是从原始序列的index*slide_step的索引开始走的
        org_index = index * self.slide_step
        if self.mode == "train":
            return np.float32(self.train[org_index:org_index + self.win_size]), \
                np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[org_index:org_index + self.win_size]), \
                np.float32(self.test_labels[self.test_labels[0:self.win_size]])
        elif self.mode == 'test':
            return np.float32(self.val[org_index:org_index + self.win_size]), \
                np.float32(self.test_labels[org_index:org_index + self.win_size])
        else:
            return np.float32(self.val[
                              org_index // self.slide_step * self.win_size:org_index // self.slide_step * self.win_size + self.win_size]), np.float32(
                self.test_labels[org_index // self.slide_step * self.win_size:org_index // self.slide_step * self.win_size + self.win_size])




def get_data_loader(data_path, batch_size, win_size=100, slide_step=100, mode='train',
                    transform=True, dataset='KDD'):
    if (dataset == 'SMD'):
        dataset = SMDSeqDataset(data_path, win_size, slide_step, mode, transform)
    elif (dataset == 'MSL'):
        dataset = MSLSeqDataset(data_path, win_size, slide_step, mode, transform)
    elif (dataset == 'SMAP'):
        dataset = SMAPSeqDataset(data_path, win_size, slide_step, mode, transform)
    elif (dataset == 'PSM'):
        dataset = PSMSeqDataset(data_path, win_size, slide_step, mode, transform)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader




#-------------------------异常检测的Dataset总类--------------------------
# 参数有:win_size=100, slide_step=100, mode='train', transform=True
class AnomalyDataset:
    def __init__(self, win_size=100, slide_step=100, mode='train', transform=True):
        self.win_size = win_size
        self.slide_step = slide_step
        self.mode = mode
        self.transform = transform

    def MSLDataset(self, data_path='./data/MSL'):
        return MSLSeqDataset(data_path, self.win_size, self.slide_step, self.transform)

    def PSMDataset(self, data_path='./data/PSM'):
        return PSMSeqDataset(data_path, self.win_size, self.slide_step, self.transform)

    def SMAPDataset(self, data_path='./data/SMAP'):
        return SMAPSeqDataset(data_path, self.win_size, self.slide_step, self.transform)

    def SMDDataset(self, data_path='./data/SMD'):
        return SMDSeqDataset(data_path, self.win_size, self.slide_step, self.transform)


if __name__ == '__main__':
    print('--------------MSL数据集------------------')
    msl_data_loader = AnomalyDataset().MSLDataset()