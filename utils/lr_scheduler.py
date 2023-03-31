import math
from functools import partial

class LRScheduler:
    def __init__(self, name, lr, iters_per_epoch, total_epochs, **kwargs):
        """
        Supported lr schedulers: [cos, warmcos, multistep]

        Args:
            lr (float): learning rate.
            iters_per_peoch (int): number of iterations in one epoch.
            total_epochs (int): number of epochs in training.
            kwargs (dict):
                - cos: None
                - warmcos: [warmup_epochs, warmup_lr_start (default 1e-6)]
                - multistep: [milestones (epochs), gamma (default 0.1)]
        """

        self.lr = lr
        self.iters_per_epoch = iters_per_epoch
        self.total_epochs = total_epochs
        self.total_iters = iters_per_epoch * total_epochs

        self.__dict__.update(kwargs)

        self.lr_func = self._get_lr_func(name)

    def update_lr(self, iters):
        return self.lr_func(iters)

    def _get_lr_func(self, name): # 余弦学习率
        if name == "cos":  # cosine lr schedule
            lr_func = partial(cos_lr, self.lr, self.total_iters)
        elif name == "warmcos": # 预热余弦学习率
            warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
            warmup_lr_start = getattr(self, "warmup_lr_start", 1e-6)
            lr_func = partial(
                warm_cos_lr,
                self.lr,
                self.total_iters,
                warmup_total_iters,
                warmup_lr_start,
            )
        else:
            raise ValueError("Scheduler version {} not supported.".format(name))
        return lr_func


def cos_lr(lr, total_iters, iters):
    """Cosine learning rate"""
    lr *= 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))
    return lr

def warm_cos_lr(lr, total_iters, warmup_total_iters, warmup_lr_start, iters):
    """Cosine learning rate with warm up."""
    if iters <= warmup_total_iters:
        lr = (lr - warmup_lr_start) * iters / float(
            warmup_total_iters
        ) + warmup_lr_start
    else:
        lr *= 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters)
            )
        )
    return lr



