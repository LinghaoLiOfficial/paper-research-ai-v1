import torch.nn as nn

from config.model.LossFunctionNameStr import LossFunctionNameStr
from entity.loss_function.FocalLoss import FocalLoss
from entity.loss_function.FocalRecallLoss import FocalRecallLoss
from entity.loss_function.PrecisionRecallTradeoffLoss import PrecisionRecallTradeoffLoss
from entity.loss_function.RecallLoss import RecallLoss


class LossFunctionFactory:
    @classmethod
    def get_loss_function(cls, loss_function_name, hyper_params: dict = {}):
        loss_function_mapping = {
            LossFunctionNameStr.CrossEntropyLoss: lambda: nn.CrossEntropyLoss(),
            LossFunctionNameStr.MSELoss: lambda: nn.MSELoss(),
            LossFunctionNameStr.FocalLoss: lambda: FocalLoss(),
            LossFunctionNameStr.RecallLoss: lambda: RecallLoss(),
            LossFunctionNameStr.PrecisionRecallTradeoffLoss: lambda: PrecisionRecallTradeoffLoss(),
            LossFunctionNameStr.FocalRecallLoss: lambda: FocalRecallLoss(),
        }

        loss_function_constructor = loss_function_mapping.get(loss_function_name, None)
        if loss_function_constructor is None:
            return None

        # 调用lambda模板实现延迟实例化
        return loss_function_constructor()

