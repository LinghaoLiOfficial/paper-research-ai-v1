# 配置超参数优化方法英文名
class ParamMethodLabelStr:
    ALL_MANUAL = "all_manual"
    HALF_AUTO = "half_auto"

    @classmethod
    def get_all(cls):
        return [v for k, v in cls.__dict__.items() if isinstance(v, str) and (not (k.startswith("__") and k.endswith("__")))]


