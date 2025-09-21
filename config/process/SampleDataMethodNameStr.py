from config.process.SampleDataMethodLabelStr import SampleDataMethodLabelStr


# 数据采样方法中文名
class SampleDataMethodNameStr:
    RANDOM_UNDER = "随机降采样"
    SMOTE_OVER = "SMOTE过采样"

    @classmethod
    def mapping(cls):
        return {
            SampleDataMethodLabelStr.RANDOM_UNDER: cls.RANDOM_UNDER,
            SampleDataMethodLabelStr.SMOTE_OVER: cls.SMOTE_OVER,
        }
