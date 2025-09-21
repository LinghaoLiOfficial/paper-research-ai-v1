# 处理目标值方法英文名
class TargetValueMethodLabelStr:
    ALL_BINARIZE = "all_binarize"

    @classmethod
    def get_all(cls):
        return [v for k, v in cls.__dict__.items() if isinstance(v, str) and (not (k.startswith("__") and k.endswith("__")))]


