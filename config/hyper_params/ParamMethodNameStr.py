from config.hyper_params.ParamMethodLabelStr import ParamMethodLabelStr


# 配置超参数优化方法中文名
class ParamMethodNameStr:
    ALL_MANUAL = "全手动配置【无超参数优化】"
    HALF_AUTO = "半自动配置【部分超参数优化】"

    @classmethod
    def mapping(cls):
        return {
            ParamMethodLabelStr.ALL_MANUAL: cls.ALL_MANUAL,
            ParamMethodLabelStr.HALF_AUTO: cls.HALF_AUTO,
        }
