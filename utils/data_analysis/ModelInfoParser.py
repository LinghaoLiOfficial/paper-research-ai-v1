from config.model.ModelTypeLabelStr import ModelTypeLabelStr


class ModelInfoParser:
    @classmethod
    def get(cls, model_name, model_type):

        if model_type == ModelTypeLabelStr.DEEP_LEARNING_CONTRASTIVE_SELF_SUPERVISED_MODEL:

            return [
                {
                    "id": 0,
                    "name": "负样本重构模型",
                    "value": "negative_model",
                    "children": [
                        {
                            "id": 0,
                            "name": "自编码器",
                            "value": "AutoEncoder",
                            "param": [
                                {
                                    "id": 0,
                                    "zhName": "hidden_dim2",
                                    "enName": "hidden_dim2",
                                },
                                {
                                    "id": 1,
                                    "zhName": "hidden_dim3",
                                    "enName": "hidden_dim3",
                                },
                                {
                                    "id": 2,
                                    "zhName": "hidden_dim4",
                                    "enName": "hidden_dim4",
                                }
                            ]
                        },
                        {
                            "id": 1,
                            "name": "变分自编码器",
                            "value": "VAE",
                            "param": [
                                {
                                    "id": 0,
                                    "zhName": "hidden_dim2",
                                    "enName": "hidden_dim2",
                                },
                                {
                                    "id": 1,
                                    "zhName": "hidden_dim3",
                                    "enName": "hidden_dim3",
                                },
                                {
                                    "id": 2,
                                    "zhName": "hidden_dim4",
                                    "enName": "hidden_dim4",
                                }
                            ]
                        },
                        {
                            "id": 2,
                            "name": "循环自编码器",
                            "value": "RAE",
                            "param": [
                                {
                                    "id": 0,
                                    "zhName": "hidden_dim2",
                                    "enName": "hidden_dim2",
                                },
                                {
                                    "id": 1,
                                    "zhName": "hidden_dim3",
                                    "enName": "hidden_dim3",
                                },
                                {
                                    "id": 2,
                                    "zhName": "hidden_dim4",
                                    "enName": "hidden_dim4",
                                }
                            ]
                        },
                        {
                            "id": 3,
                            "name": "循环-变分自编码器",
                            "value": "RVAE",
                            "param": [
                                {
                                    "id": 0,
                                    "zhName": "hidden_dim2",
                                    "enName": "hidden_dim2",
                                },
                                {
                                    "id": 1,
                                    "zhName": "hidden_dim3",
                                    "enName": "hidden_dim3",
                                },
                                {
                                    "id": 2,
                                    "zhName": "hidden_dim4",
                                    "enName": "hidden_dim4",
                                }
                            ]
                        },
                        {
                            "id": 4,
                            "name": "注意力变分自编码器",
                            "value": "AttentionVAE",
                            "param": [
                                {
                                    "id": 0,
                                    "zhName": "hidden_dim2",
                                    "enName": "hidden_dim2",
                                },
                                {
                                    "id": 1,
                                    "zhName": "hidden_dim3",
                                    "enName": "hidden_dim3",
                                },
                                {
                                    "id": 2,
                                    "zhName": "hidden_dim4",
                                    "enName": "hidden_dim4",
                                }
                            ]
                        }
                    ]
                },

                {
                    "id": 1,
                    "name": "正样本重构模型",
                    "value": "positive_model",
                    "children": [
                        {
                            "id": 0,
                            "name": "贝叶斯多元回归",
                            "value": "BayesianRegression",
                            "param": [
                                {
                                    "id": 0,
                                    "zhName": "alpha",
                                    "enName": "alpha",
                                },
                                {
                                    "id": 1,
                                    "zhName": "beta",
                                    "enName": "beta",
                                }
                            ]
                        },
                        {
                            "id": 1,
                            "name": "自编码器",
                            "value": "AutoEncoder",
                            "param": [
                                {
                                    "id": 0,
                                    "zhName": "hidden_dim2",
                                    "enName": "hidden_dim2",
                                },
                                {
                                    "id": 1,
                                    "zhName": "hidden_dim3",
                                    "enName": "hidden_dim3",
                                },
                                {
                                    "id": 2,
                                    "zhName": "hidden_dim4",
                                    "enName": "hidden_dim4",
                                }
                            ]
                        },
                        {
                            "id": 2,
                            "name": "变分自编码器",
                            "value": "VAE",
                            "param": [
                                {
                                    "id": 0,
                                    "zhName": "hidden_dim2",
                                    "enName": "hidden_dim2",
                                },
                                {
                                    "id": 1,
                                    "zhName": "hidden_dim3",
                                    "enName": "hidden_dim3",
                                },
                                {
                                    "id": 2,
                                    "zhName": "hidden_dim4",
                                    "enName": "hidden_dim4",
                                }
                            ]
                        },
                        {
                            "id": 3,
                            "name": "循环自编码器",
                            "value": "RAE",
                            "param": [
                                {
                                    "id": 0,
                                    "zhName": "hidden_dim2",
                                    "enName": "hidden_dim2",
                                },
                                {
                                    "id": 1,
                                    "zhName": "hidden_dim3",
                                    "enName": "hidden_dim3",
                                },
                                {
                                    "id": 2,
                                    "zhName": "hidden_dim4",
                                    "enName": "hidden_dim4",
                                }
                            ]
                        },
                        {
                            "id": 4,
                            "name": "循环-变分自编码器",
                            "value": "RVAE",
                            "param": [
                                {
                                    "id": 0,
                                    "zhName": "hidden_dim2",
                                    "enName": "hidden_dim2",
                                },
                                {
                                    "id": 1,
                                    "zhName": "hidden_dim3",
                                    "enName": "hidden_dim3",
                                },
                                {
                                    "id": 2,
                                    "zhName": "hidden_dim4",
                                    "enName": "hidden_dim4",
                                }
                            ]
                        },
                        {
                            "id": 5,
                            "name": "注意力变分自编码器",
                            "value": "AttentionVAE",
                            "param": [
                                {
                                    "id": 0,
                                    "zhName": "hidden_dim2",
                                    "enName": "hidden_dim2",
                                },
                                {
                                    "id": 1,
                                    "zhName": "hidden_dim3",
                                    "enName": "hidden_dim3",
                                },
                                {
                                    "id": 2,
                                    "zhName": "hidden_dim4",
                                    "enName": "hidden_dim4",
                                }
                            ]
                        }
                    ]
                },

                {
                    "id": 2,
                    "name": "阈值设定",
                    "value": "threshold_setting",
                    "children": [
                        {
                            "id": 0,
                            "name": "自适应阈值调整",
                            "value": "adaptive_threshold_adjustment",
                            "param": [
                                {
                                    "id": 0,
                                    "zhName": "超阈值特征比例",
                                    "enName": "feature_threshold_ratio"
                                }
                            ]
                        },
                        {
                            "id": 1,
                            "name": "固定阈值设定",
                            "value": "solid_threshold_setting",
                            "param": [
                                {
                                    "id": 0,
                                    "zhName": "阈值大小",
                                    "enName": "threshold_value",
                                }
                            ]
                        }
                    ]
                }
            ]

        elif model_type == ModelTypeLabelStr.DEEP_LEARNING_SELF_SUPERVISED_MODEL:

            return [
                {
                    "id": 0,
                    "name": "模型",
                    "value": "model",
                    "children": [
                        {
                            "id": 0,
                            "name": "自身",
                            "value": "self",
                            "param": [
                                {
                                    "id": 0,
                                    "zhName": "hidden_dim2",
                                    "enName": "hidden_dim2",
                                },
                                {
                                    "id": 1,
                                    "zhName": "hidden_dim3",
                                    "enName": "hidden_dim3",
                                },
                                {
                                    "id": 2,
                                    "zhName": "hidden_dim4",
                                    "enName": "hidden_dim4",
                                }
                            ]
                        }
                    ]
                },

                {
                    "id": 1,
                    "name": "阈值设定",
                    "value": "threshold_setting",
                    "children": [
                        {
                            "id": 0,
                            "name": "固定阈值设定",
                            "value": "solid_threshold_setting",
                            "param": [
                                {
                                    "id": 0,
                                    "zhName": "阈值大小",
                                    "enName": "threshold_value",
                                }
                            ]
                        }
                    ]
                }
            ]

        elif model_type == ModelTypeLabelStr.DEEP_LEARNING_SUPERVISED_MODEL:

            return [
                {
                    "id": 0,
                    "name": "模型",
                    "value": "model",
                    "children": [
                        {
                            "id": 0,
                            "name": "自身",
                            "value": "self",
                            "param": [
                                {
                                    "id": 0,
                                    "zhName": "hidden_dim",
                                    "enName": "hidden_dim",
                                }
                            ]
                        }
                    ]
                }
            ]

        elif model_type == ModelTypeLabelStr.MACHINE_LEARNING_SUPERVISED_MODEL:

            return [
                {
                    "id": 0,
                    "name": "模型",
                    "value": "model",
                    "children": [
                        {
                            "id": 0,
                            "name": "自身",
                            "value": "self",
                            "param": [
                                {
                                    "id": 0,
                                    "zhName": "hidden_dim",
                                    "enName": "hidden_dim",
                                }
                            ]
                        }
                    ]
                }
            ]