from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmupQuadraticLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.last_epoch = last_epoch
        super(WarmupQuadraticLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            progress = (self.total_epochs - self.last_epoch) / (self.total_epochs - self.warmup_epochs)
            return [base_lr * progress**2 for base_lr in self.base_lrs]

class StableQuadraticLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.last_epoch = last_epoch
        super(StableQuadraticLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return self.base_lrs
        else:
            progress = (self.total_epochs - self.last_epoch) / (self.total_epochs - self.warmup_epochs)
            return [base_lr * progress**2 for base_lr in self.base_lrs]

