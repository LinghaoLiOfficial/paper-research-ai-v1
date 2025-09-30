import torch

from config.model.OptimizerNameStr import OptimizerNameStr


class OptimizerFactory:
    @classmethod
    def get_optimizer(cls, optimizer_name, model, hyper_params: dict = {}):
        optimizer_mapping = {
            OptimizerNameStr.Adam: lambda: torch.optim.Adam(
                model.parameters(),
                lr=hyper_params["lr"],
                weight_decay=hyper_params["weight_decay"]
            ),
        }

        optimizer_constructor = optimizer_mapping.get(optimizer_name, None)
        if optimizer_constructor is None:
            return None

        # 调用lambda模板实现延迟实例化
        return optimizer_constructor()