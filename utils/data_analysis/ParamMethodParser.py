from config.hyper_params.ParamMethodLabelStr import ParamMethodLabelStr
from config.hyper_params.ParamMethodNameStr import ParamMethodNameStr


class ParamMethodParser:
    @classmethod
    def get(cls):
        param_method_list = ParamMethodLabelStr.get_all()

        out_list = []
        for i, method in enumerate(param_method_list):
            out_list.append({
                "id": i,
                "name": ParamMethodNameStr.mapping()[method],
                "value": method
            })

        return out_list
