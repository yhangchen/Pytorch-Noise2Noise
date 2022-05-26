import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

class HDRLoss(nn.Module):

    def __init__(self, eps=0.01):
        super(HDRLoss, self).__init__()
        self.eps = eps

    def forward(self, denoised, target):
        loss = ((denoised - target)**2) / (denoised + self.eps)**2
        return torch.mean(loss.view(-1))


class RampedLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 max_epoch,
                 ramp_down_percent=0.3,
                 last_epoch=-1,
                 verbose=False):
        self.ramp_down_percent = ramp_down_percent
        self.max_epoch = max_epoch
        super(RampedLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            raise AssertionError(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.")

        return [
            self.ramped_down(group['initial_lr'])
            for group in self.optimizer.param_groups
        ]

    def ramped_down(self, learning_rate):
        ramp_down_start = self.max_epoch * (1 - self.ramp_down_percent)
        if self.last_epoch >= ramp_down_start:
            t = (self.last_epoch - ramp_down_start) / \
                self.ramp_down_percent / self.max_epoch
            smooth = (0.5 + np.cos(t * np.pi) / 2)**2
            return learning_rate * smooth
        return learning_rate


def psnr(denoised, ground_truth):
    mse = torch.mean((denoised - ground_truth)**2)
    return -10 * torch.log10(mse + 10**-8)
