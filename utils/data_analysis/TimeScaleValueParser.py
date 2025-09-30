from config.process.TimeScaleLabelStr import TimeScaleLabelStr
from config.process.TimeScaleNameStr import TimeScaleNameStr


class TimeScaleValueParser:
    @classmethod
    def get(cls):
        param_method_list = TimeScaleLabelStr.get_all()

        out_list = []
        for i, method in enumerate(param_method_list):
            out_list.append({
                "id": i,
                "name": TimeScaleNameStr.mapping()[method],
                "value": method
            })

        return out_list
