# 超参数英文名
class ParamLabelStr:
    EPOCHS = "epochs"
    LR = "lr"
    TRAIN_RATIO = "train_ratio"
    BATCH_SIZE = "batch_size"
    PREDICT_RANGE = "predict_range"
    RANDOM_SEED = "random_seed"
    TIMESTEP = "timestep"

    @classmethod
    def get_all(cls):
        return [v for k, v in cls.__dict__.items() if isinstance(v, str) and (not (k.startswith("__") and k.endswith("__")))]


