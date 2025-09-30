from config.process.TargetValueMethodLabelStr import TargetValueMethodLabelStr


# 处理目标值方法中文名
class TargetValueMethodNameStr:
    ALL_BINARIZE = "所有标签二值化"

    @classmethod
    def mapping(cls):
        return {
            TargetValueMethodLabelStr.ALL_BINARIZE: cls.ALL_BINARIZE,
        }
