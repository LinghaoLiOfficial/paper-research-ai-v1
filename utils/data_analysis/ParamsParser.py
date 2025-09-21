from config.hyper_params.ParamLabelStr import ParamLabelStr
from config.hyper_params.ParamNameStr import ParamZHNameStr
from utils.data_analysis.HyperParamsParser import HyperParamsParser


class ParamsParser:
    @classmethod
    def get(cls):
        param_list = ParamLabelStr.get_all()

        hyper_params_value_dict = HyperParamsParser.load()
        # filtered_hyper_params_value_dict = {k: v for k, v in hyper_params_value_dict.items() if k in param_list}

        out_list = []
        for i, param in enumerate(param_list):
            out_list.append({
                "id": i,
                "enName": param,
                "zhName": ParamZHNameStr.mapping()[param],
                # "value": filtered_hyper_params_value_dict[param]
            })

        return out_list
