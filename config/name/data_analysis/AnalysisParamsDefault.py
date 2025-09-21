from config.name.BaseName import BaseName


class AnalysisParamsDefault(BaseName):
    EPOCHS = 20
    LR = 1e-4
    TRAIN_RATIO = 0.8
    BATCH_SIZE = 64
    PREDICT_RANGE = 1
    RANDOM_SEED = 56
    TIMESTEP = 40
    WEIGHT_DECAY = 0.001
