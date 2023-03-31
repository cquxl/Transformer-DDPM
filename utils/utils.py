import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch import distributed as dist

_LOCAL_PROCESS_GROUP = None

def to_var(x, volatile=False):
    """
    Varibale变量，可以反向传播，涉及到计算图的计算
    requires_grad：variable默认是不需要被求导的，即requires_grad属性默认为False
    variable的volatile属性默认为False，如果某一个variable的volatile属性被设为True，不会求导
    retain_graph：单次反向传播后，计算图会free掉，不会累积，retain_graph=True来保存这些缓存。
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def str2bool(v): # True or False
    return v.lower() in ('true')

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if _LOCAL_PROCESS_GROUP is None:
        return get_rank() #0

    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

if __name__ == '__main__':
    print(get_local_rank())
