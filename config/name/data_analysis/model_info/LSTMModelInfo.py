from config.name.BaseName import BaseName
from config.name.data_analysis.AnalysisModelEnName import AnalysisModelEnName


class LSTMModelInfo(BaseName):
    MODEL = {
        "id": 0,
        "label": "模型",
        "name": "model",
        "children": [
            {
                "id": 0,
                "label": AnalysisModelEnName.LSTM,
                "name": AnalysisModelEnName.LSTM,
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