import math

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class DecayingCosineAnnealingWarmRestarts(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, gamma=0.9):
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)
        self.gamma = gamma

    def get_lr(self):
        # Raise warning about inconsistent calls to the scheduler.
        super().get_lr()
        return [
            self.eta_min
            + (base_lr * self.gamma**self.last_epoch - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]
