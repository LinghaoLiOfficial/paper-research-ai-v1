import torch


class LRScheduler:

    @classmethod
    def update_lr(cls, scheduler):

        scheduler.step()

    @classmethod
    def load_scheduler(cls, scheduler_name: str, optimizer, hyper_params: dict):

        if scheduler_name == "StepLR":
            optimizer = cls._load_StepLR(optimizer, hyper_params)

        elif scheduler_name == "ExponentialLR":
            optimizer = cls._load_ExponentialLR(optimizer, hyper_params)

        else:
            optimizer = cls._load_StepLR(optimizer, hyper_params)

        return optimizer

    @classmethod
    def _load_StepLR(cls, optimizer, hyper_params: dict):

        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    @classmethod
    def _load_ExponentialLR(cls, optimizer, hyper_params: dict):

        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hyper_params["gamma"])
