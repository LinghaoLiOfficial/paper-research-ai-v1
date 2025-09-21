from config.process.NullValueMethodLabelStr import NullValueMethodLabelStr


# 处理缺失值方法中文名
class NullValueMethodNameStr:
    FILL_NULL = "填充缺失值"
    DEL_NULL = "删除缺失值"

    @classmethod
    def mapping(cls):
        return {
            NullValueMethodLabelStr.FILL_NULL: cls.FILL_NULL,
            NullValueMethodLabelStr.DEL_NULL: cls.DEL_NULL,
        }
