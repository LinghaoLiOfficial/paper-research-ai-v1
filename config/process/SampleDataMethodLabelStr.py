# 数据采样方法英文名
class SampleDataMethodLabelStr:
    RANDOM_UNDER = "random_under"
    SMOTE_OVER = "smote_over"

    @classmethod
    def get_all(cls):
        return [v for k, v in cls.__dict__.items() if isinstance(v, str) and (not (k.startswith("__") and k.endswith("__")))]


