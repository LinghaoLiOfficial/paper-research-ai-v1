from config.process.SampleDataMethodLabelStr import SampleDataMethodLabelStr
from config.process.SampleDataMethodNameStr import SampleDataMethodNameStr


class SampleDataMethodParser:
    @classmethod
    def get(cls):
        param_method_list = SampleDataMethodLabelStr.get_all()

        out_list = []
        for i, method in enumerate(param_method_list):
            out_list.append({
                "id": i,
                "name": SampleDataMethodNameStr.mapping()[method],
                "value": method
            })

        return out_list
