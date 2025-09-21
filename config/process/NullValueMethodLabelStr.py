# 处理缺失值方法英文名
class NullValueMethodLabelStr:
    FILL_NULL = "fill_null"
    DEL_NULL = "del_null"

    @classmethod
    def get_all(cls):
        return [v for k, v in cls.__dict__.items() if isinstance(v, str) and (not (k.startswith("__") and k.endswith("__")))]


