# 时序聚合方法英文名
class TimeScaleLabelStr:
    TIME_SCALE_1 = "time_scale_1"
    TIME_SCALE_6 = "time_scale_6"
    TIME_SCALE_36 = "time_scale_36"
    TIME_SCALE_360 = "time_scale_360"

    @classmethod
    def get_all(cls):
        return [v for k, v in cls.__dict__.items() if isinstance(v, str) and (not (k.startswith("__") and k.endswith("__")))]


