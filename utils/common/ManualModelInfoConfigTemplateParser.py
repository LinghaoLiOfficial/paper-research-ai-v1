from config.model.ModelTypeLabelStr import ModelTypeLabelStr


class ManualModelInfoConfigTemplateParser:
    @classmethod
    def generate_blank_template(cls, model_name, model_type):
        BLANK_DEFAULT_VALUE = None

        if model_type == ModelTypeLabelStr.DEEP_LEARNING_CONTRASTIVE_SELF_SUPERVISED_MODEL:

            return {
                "negative_model": {
                    "method": "AttentionVAE",
                    "data": {
                        "AutoEncoder": {
                            "hidden_dim2": 24,
                            "hidden_dim3": 12,
                            "hidden_dim4": 3
                        },
                        "RAE": {
                            "hidden_dim2": 24,
                            "hidden_dim3": 12,
                            "hidden_dim4": 3
                        },
                        "VAE": {
                            "hidden_dim2": 24,
                            "hidden_dim3": 12,
                            "hidden_dim4": 3
                        },
                        "RVAE": {
                            "hidden_dim2": 24,
                            "hidden_dim3": 12,
                            "hidden_dim4": 3
                        },
                        "AttentionVAE": {
                            "hidden_dim2": 24,
                            "hidden_dim3": 12,
                            "hidden_dim4": 3
                        }
                    }
                },

                "positive_model": {
                    "method": "BayesianRegression",
                    "data": {
                        "BayesianRegression": {
                            "alpha": 1,
                            "beta": 1
                        },
                        "AutoEncoder": {
                            "hidden_dim2": 24,
                            "hidden_dim3": 12,
                            "hidden_dim4": 3
                        },
                        "RAE": {
                            "hidden_dim2": 24,
                            "hidden_dim3": 12,
                            "hidden_dim4": 6
                        },
                        "VAE": {
                            "hidden_dim2": 24,
                            "hidden_dim3": 12,
                            "hidden_dim4": 6
                        },
                        "RVAE": {
                            "hidden_dim2": 24,
                            "hidden_dim3": 12,
                            "hidden_dim4": 6
                        },
                        "AttentionVAE": {
                            "hidden_dim2": 24,
                            "hidden_dim3": 12,
                            "hidden_dim4": 6
                        }
                    }
                },

                "threshold_setting": {
                    "method": "adaptive_threshold_adjustment",
                    "data": {
                        "adaptive_threshold_adjustment": {
                            "feature_threshold_ratio": 0.5
                        },
                        "solid_threshold_setting": {
                            "threshold_value": 0.5
                        }
                    }
                }
            }

        elif model_type == ModelTypeLabelStr.DEEP_LEARNING_SELF_SUPERVISED_MODEL:

            return {
                "model": {
                    "method": "self",
                    "data": {
                        "self": {
                            "hidden_dim2": 24,
                            "hidden_dim3": 12,
                            "hidden_dim4": 3
                        }
                    }
                },

                "threshold_setting": {
                    "method": "solid_threshold_setting",
                    "data": {
                        "solid_threshold_setting": {
                            "threshold_value": 0.5
                        }
                    }
                }
            }

        elif model_type == ModelTypeLabelStr.DEEP_LEARNING_SUPERVISED_MODEL:

            return {
                "model": {
                    "method": "self",
                    "data": {
                        "self": {
                            "hidden_dim": 24
                        }
                    }
                },
            }

        elif model_type == ModelTypeLabelStr.MACHINE_LEARNING_SUPERVISED_MODEL:

            return {
                "model": {
                    "method": "self",
                    "data": {
                        "self": {
                            "hidden_dim": 24
                        }
                    }
                },
            }

