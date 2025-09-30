# 模型类型英文名
class ModelTypeLabelStr:
    MACHINE_LEARNING_SUPERVISED_MODEL = "machine_learning_supervised_model"
    DEEP_LEARNING_SUPERVISED_MODEL = "deep_learning_supervised_model"
    DEEP_LEARNING_SELF_SUPERVISED_MODEL = "deep_learning_self_supervised_model"
    DEEP_LEARNING_CONTRASTIVE_SELF_SUPERVISED_MODEL = "deep_learning_contrastive_self_supervised_model"

    @classmethod
    def get_all(cls):
        return [v for k, v in cls.__dict__.items() if isinstance(v, str) and (not (k.startswith("__") and k.endswith("__")))]


