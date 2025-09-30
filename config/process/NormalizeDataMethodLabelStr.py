#数据标准化方法英文名
class NormalizeDataMethodLabelStr:
    Z_SCORE = "z_score"
    MIN_MAX = "min_max"
    Z_SCORE_PLUS_MIN_MAX = "z_score_plus_min_max"

    @classmethod
    def get_all(cls):
        return [v for k, v in cls.__dict__.items() if isinstance(v, str) and (not (k.startswith("__") and k.endswith("__")))]


