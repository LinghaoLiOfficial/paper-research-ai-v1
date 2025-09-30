from config.process.NullValueMethodLabelStr import NullValueMethodLabelStr
from config.process.NullValueMethodNameStr import NullValueMethodNameStr


class NullValueMethodParser:
    @classmethod
    def get(cls):
        param_method_list = NullValueMethodLabelStr.get_all()

        out_list = []
        for i, method in enumerate(param_method_list):
            out_list.append({
                "id": i,
                "name": NullValueMethodNameStr.mapping()[method],
                "value": method
            })

        return out_list
