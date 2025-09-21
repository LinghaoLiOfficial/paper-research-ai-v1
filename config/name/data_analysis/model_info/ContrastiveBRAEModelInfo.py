from config.name.BaseName import BaseName
from config.name.data_analysis.AnalysisModelEnName import AnalysisModelEnName
from config.name.data_analysis.module.ContrastiveBRAEName import ContrastiveBRAEName


class ContrastiveBRAEModelInfo(BaseName):
    NEGATIVE_MODEL = {
        "id": 0,
        "label": "负样本重构模型",
        "name": "negative_model",
        "children": [
            {
                "id": 0,
                "label": AnalysisModelEnName.AutoEncoder,
                "name": AnalysisModelEnName.AutoEncoder,
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
            {
                "id": 1,
                "label": AnalysisModelEnName.VAE,
                "name": AnalysisModelEnName.VAE,
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
            {
                "id": 2,
                "label": AnalysisModelEnName.RAE,
                "name": AnalysisModelEnName.RAE,
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
            {
                "id": 3,
                "label": AnalysisModelEnName.RVAE,
                "name": AnalysisModelEnName.RVAE,
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
            {
                "id": 4,
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

    POSITIVE_MODEL = {
        "id": 1,
        "label": "正样本重构模型",
        "name": "positive_model",
        "children": [
            {
                "id": 0,
                "label": AnalysisModelEnName.BayesianRegression,
                "name": AnalysisModelEnName.BayesianRegression,
                "param": [
                    {
                        "id": 0,
                        "label": "alpha",
                        "name": "alpha",
                        "value": {
                            "start": 0,
                            "end": 1,
                            "type": "float"
                        }
                    },
                    {
                        "id": 1,
                        "label": "beta",
                        "name": "beta",
                        "value": {
                            "start": 0,
                            "end": 1,
                            "type": "float"
                        }
                    },
                ]
            },

            {
                "id": 1,
                "label": AnalysisModelEnName.AutoEncoder,
                "name": AnalysisModelEnName.AutoEncoder,
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
            {
                "id": 2,
                "label": AnalysisModelEnName.VAE,
                "name": AnalysisModelEnName.VAE,
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
            {
                "id": 3,
                "label": AnalysisModelEnName.RAE,
                "name": AnalysisModelEnName.RAE,
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
            {
                "id": 4,
                "label": AnalysisModelEnName.RVAE,
                "name": AnalysisModelEnName.RVAE,
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
            {
                "id": 5,
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

            {
                "id": 6,
                "label": AnalysisModelEnName.KernelBayesianRegression,
                "name": AnalysisModelEnName.KernelBayesianRegression,
                "param": [
                    {
                        "id": 0,
                        "label": "alpha",
                        "name": "alpha",
                        "value": {
                            "start": 0,
                            "end": 1,
                            "type": "float"
                        }
                    },
                    {
                        "id": 1,
                        "label": "beta",
                        "name": "beta",
                        "value": {
                            "start": 0,
                            "end": 1,
                            "type": "float"
                        }
                    },
                    {
                        "id": 2,
                        "label": "length_scale",
                        "name": "length_scale",
                        "value": {
                            "start": 0,
                            "end": 1,
                            "type": "float"
                        }
                    },
                ]
            },

        ]
    }

    THRESHOLD_SETTING = {
        "id": 2,
        "label": "阈值设定",
        "name": "threshold_setting",
        "children": [
            {
                "id": 0,
                "label": "自适应阈值调整",
                "name": ContrastiveBRAEName.ADAPTIVE_THRESHOLD_ADJUSTMENT,
                "param": [
                    {
                        "id": 0,
                        "label": "特征比例",
                        "name": "feature_threshold_ratio",
                        "value": {
                            "start": 0,
                            "end": 1,
                            "type": "float"
                        }
                    }
                ]
            },
            {
                "id": 1,
                "label": "固定阈值",
                "name": ContrastiveBRAEName.THRESHOLD_SETTING,
                "param": [
                    {
                        "id": 0,
                        "label": "阈值",
                        "name": "threshold_value",
                        "value": {
                            "start": 0,
                            "end": 1,
                            "type": "float"
                        }
                    }
                ]
            },
        ]
    }
