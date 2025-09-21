from config.name.BaseName import BaseName
from config.name.data_analysis.AnalysisModelEnName import AnalysisModelEnName


class TransformerModelInfo(BaseName):
    MODEL = {
        "id": 0,
        "label": "模型",
        "name": "model",
        "children": [
            {
                "id": 0,
                "label": AnalysisModelEnName.Transformer,
                "name": AnalysisModelEnName.Transformer,
                "param": [
                    {
                        "id": 0,
                        "label": "hidden_dim",
                        "name": "hidden_dim",
                        "value": {
                            "start": 20,
                            "end": 30,
                            "type": "int"
                        }
                    }
                ]
            },

        ]
    }