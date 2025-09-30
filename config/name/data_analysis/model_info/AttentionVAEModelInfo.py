from config.name.BaseName import BaseName
from config.name.data_analysis.AnalysisModelEnName import AnalysisModelEnName


class AttentionVAEModelInfo(BaseName):
    MODEL = {
        "id": 0,
        "label": "模型",
        "name": "model",
        "children": [
            {
                "id": 0,
                "label": AnalysisModelEnName.AttentionVAE,
                "name": AnalysisModelEnName.AttentionVAE,
                "param": [
                    {
                        "id": 0,
                        "label": "hidden_dim2",
                        "name": "hidden_dim2",
                        "value": {
                            "start": 20,
                            "end": 30,
                            "type": "int"
                        }
                    },
                    {
                        "id": 1,
                        "label": "hidden_dim3",
                        "name": "hidden_dim3",
                        "value": {
                            "start": 10,
                            "end": 20,
                            "type": "int"
                        }
                    },
                    {
                        "id": 2,
                        "label": "hidden_dim4",
                        "name": "hidden_dim4",
                        "value": {
                            "start": 3,
                            "end": 10,
                            "type": "int"
                        }
                    },
                ]
            },

        ]
    }