from config.process.SampleDataMethodLabelStr import SampleDataMethodLabelStr
from config.process.TimeScaleLabelStr import TimeScaleLabelStr


# 时序聚合方法中文法
class TimeScaleNameStr:
    TIME_SCALE_1 = "time_scale_1"
    TIME_SCALE_6 = "time_scale_6"
    TIME_SCALE_36 = "time_scale_36"
    TIME_SCALE_360 = "time_scale_360"

    @classmethod
    def mapping(cls):
        return {
            TimeScaleLabelStr.TIME_SCALE_1: cls.TIME_SCALE_1,
            TimeScaleLabelStr.TIME_SCALE_6: cls.TIME_SCALE_6,
            TimeScaleLabelStr.TIME_SCALE_36: cls.TIME_SCALE_36,
            TimeScaleLabelStr.TIME_SCALE_360: cls.TIME_SCALE_360
        }
