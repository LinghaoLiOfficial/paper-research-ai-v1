from config.hyper_params.ParamLabelStr import ParamLabelStr


# 超参数中文名
class ParamZHNameStr:
    EPOCHS = "迭代次数"
    LR = "初始学习率"
    TRAIN_RATIO = "训练集比例"
    BATCH_SIZE = "批大小"
    PREDICT_RANGE = "预测范围"
    RANDOM_SEED = "随机种子"
    TIMESTEP = "时间窗口大小"

    @classmethod
    def mapping(cls):
        return {
            ParamLabelStr.EPOCHS: cls.EPOCHS,
            ParamLabelStr.LR: cls.LR,
            ParamLabelStr.TRAIN_RATIO: cls.TRAIN_RATIO,
            ParamLabelStr.BATCH_SIZE: cls.BATCH_SIZE,
            ParamLabelStr.PREDICT_RANGE: cls.PREDICT_RANGE,
            ParamLabelStr.RANDOM_SEED: cls.RANDOM_SEED,
            ParamLabelStr.TIMESTEP: cls.TIMESTEP
        }
