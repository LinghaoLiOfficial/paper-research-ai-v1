from config.process.NormalizeDataMethodLabelStr import NormalizeDataMethodLabelStr


# 数据标准化方法中文名
class NormalizeDataMethodNameStr:
    Z_SCORE = "z-score 标准化"
    MIN_MAX = "min-max 标准化"
    Z_SCORE_PLUS_MIN_MAX = "z-score 标准化 + min-max 标准化"

    @classmethod
    def mapping(cls):
        return {
            NormalizeDataMethodLabelStr.Z_SCORE: cls.Z_SCORE,
            NormalizeDataMethodLabelStr.MIN_MAX: cls.MIN_MAX,
            NormalizeDataMethodLabelStr.Z_SCORE_PLUS_MIN_MAX: cls.Z_SCORE_PLUS_MIN_MAX
        }
