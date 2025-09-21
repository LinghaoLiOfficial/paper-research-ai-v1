from config.process.NormalizeDataMethodLabelStr import NormalizeDataMethodLabelStr
from config.process.NormalizeDataMethodNameStr import NormalizeDataMethodNameStr


class NormalizeDataMethodParser:
    @classmethod
    def get(cls):
        param_method_list = NormalizeDataMethodLabelStr.get_all()

        out_list = []
        for i, method in enumerate(param_method_list):
            out_list.append({
                "id": i,
                "name": NormalizeDataMethodNameStr.mapping()[method],
                "value": method
            })

        return out_list
