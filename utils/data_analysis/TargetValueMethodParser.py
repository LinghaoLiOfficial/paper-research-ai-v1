from config.process.TargetValueMethodLabelStr import TargetValueMethodLabelStr
from config.process.TargetValueMethodNameStr import TargetValueMethodNameStr


class TargetValueMethodParser:
    @classmethod
    def get(cls):
        param_method_list = TargetValueMethodLabelStr.get_all()

        out_list = []
        for i, method in enumerate(param_method_list):
            out_list.append({
                "id": i,
                "name": TargetValueMethodNameStr.mapping()[method],
                "value": method
            })

        return out_list
