from config.model.ModelTypeLabelStr import ModelTypeLabelStr


class HalfAutoModelInfoConfigTemplateParser:
    @classmethod
    def generate_blank_template(cls, model_name, model_type):
        BLANK_DEFAULT_VALUE = None

        if model_type == ModelTypeLabelStr.DEEP_LEARNING_CONTRASTIVE_SELF_SUPERVISED_MODEL:

            return {
                "negative_model": {
                    "method": "AttentionVAE",
                    "data": {
                        "AutoEncoder": {
                            "hidden_dim2": {
                                "min": 21,
                                "max": 30,
                                "type": "int"
                            },
                            "hidden_dim3": {
                                "min": 11,
                                "max": 20,
                                "type": "int"
                            },
                            "hidden_dim4": {
                                "min": 1,
                                "max": 10,
                                "type": "int"
                            }
                        },
                        "VAE": {
                            "hidden_dim2": {
                                "min": 21,
                                "max": 30,
                                "type": "int"
                            },
                            "hidden_dim3": {
                                "min": 11,
                                "max": 20,
                                "type": "int"
                            },
                            "hidden_dim4": {
                                "min": 1,
                                "max": 10,
                                "type": "int"
                            }
                        },
                        "RAE": {
                            "hidden_dim2": {
                                "min": 21,
                                "max": 30,
                                "type": "int"
                            },
                            "hidden_dim3": {
                                "min": 11,
                                "max": 20,
                                "type": "int"
                            },
                            "hidden_dim4": {
                                "min": 1,
                                "max": 10,
                                "type": "int"
                            }
                        },
                        "RVAE": {
                            "hidden_dim2": {
                                "min": 21,
                                "max": 30,
                                "type": "int"
                            },
                            "hidden_dim3": {
                                "min": 11,
                                "max": 20,
                                "type": "int"
                            },
                            "hidden_dim4": {
                                "min": 1,
                                "max": 10,
                                "type": "int"
                            }
                        },
                        "AttentionVAE": {
                            "hidden_dim2": {
                                "min": 41,
                                "max": 80,
                                "type": "int"
                            },
                            "hidden_dim3": {
                                "min": 11,
                                "max": 40,
                                "type": "int"
                            },
                            "hidden_dim4": {
                                "min": 1,
                                "max": 10,
                                "type": "int"
                            }
                        },
                    }
                },

                "positive_model": {
                    "method": "BayesianRegression",
                    "data": {
                        "BayesianRegression": {
                            "alpha": {
                                "min": 0.1,
                                "max": 10,
                                "type": "float"
                            },
                            "beta": {
                                "min": 0.1,
                                "max": 10,
                                "type": "float"
                            }
                        },
                        "AutoEncoder": {
                            "hidden_dim2": {
                                "min": 21,
                                "max": 30,
                                "type": "int"
                            },
                            "hidden_dim3": {
                                "min": 11,
                                "max": 20,
                                "type": "int"
                            },
                            "hidden_dim4": {
                                "min": 1,
                                "max": 10,
                                "type": "int"
                            }
                        },
                        "RAE": {
                            "hidden_dim2": {
                                "min": 21,
                                "max": 30,
                                "type": "int"
                            },
                            "hidden_dim3": {
                                "min": 11,
                                "max": 20,
                                "type": "int"
                            },
                            "hidden_dim4": {
                                "min": 1,
                                "max": 10,
                                "type": "int"
                            }
                        },
                        "VAE": {
                            "hidden_dim2": {
                                "min": 21,
                                "max": 30,
                                "type": "int"
                            },
                            "hidden_dim3": {
                                "min": 11,
                                "max": 20,
                                "type": "int"
                            },
                            "hidden_dim4": {
                                "min": 1,
                                "max": 10,
                                "type": "int"
                            }
                        },
                        "RVAE": {
                            "hidden_dim2": {
                                "min": 21,
                                "max": 30,
                                "type": "int"
                            },
                            "hidden_dim3": {
                                "min": 11,
                                "max": 20,
                                "type": "int"
                            },
                            "hidden_dim4": {
                                "min": 1,
                                "max": 10,
                                "type": "int"
                            }
                        },
                        "AttentionVAE": {
                            "hidden_dim2": {
                                "min": 21,
                                "max": 30,
                                "type": "int"
                            },
                            "hidden_dim3": {
                                "min": 11,
                                "max": 20,
                                "type": "int"
                            },
                            "hidden_dim4": {
                                "min": 1,
                                "max": 10,
                                "type": "int"
                            }
                        },
                    }
                },

                "threshold_setting": {
                    "method": "adaptive_threshold_adjustment",
                    "data": {
                        "adaptive_threshold_adjustment": {
                            "feature_threshold_ratio": 1
                        },
                        "solid_threshold_setting": {
                            "threshold_value": {
                                "min": 0.1,
                                "max": 1,
                                "type": "float"
                            }
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
                            "hidden_dim2": {
                                "min": 21,
                                "max": 30,
                                "type": "int"
                            },
                            "hidden_dim3": {
                                "min": 11,
                                "max": 20,
                                "type": "int"
                            },
                            "hidden_dim4": {
                                "min": 1,
                                "max": 10,
                                "type": "int"
                            }
                        }
                    }
                },

                # "threshold_setting": {
                #     "method": "solid_threshold_setting",
                #     "data": {
                #         "solid_threshold_setting": {
                #             "threshold_value": {
                #                 "min": 0.1,
                #                 "max": 1,
                #                 "type": "float"
                #             }
                #         }
                #     }
                # }
            }

        elif model_type == ModelTypeLabelStr.DEEP_LEARNING_SUPERVISED_MODEL:

            return {
                "model": {
                    "method": "self",
                    "data": {
                        "self": {
                            "hidden_dim": {
                                "min": 3,
                                "max": 30,
                                "type": "int"
                            }
                        }
                    }
                }
            }

        elif model_type == ModelTypeLabelStr.MACHINE_LEARNING_SUPERVISED_MODEL:

            return {
                "model": {
                    "method": "self",
                    "data": {
                        "self": {
                            "hidden_dim": {
                                "min": 3,
                                "max": 30,
                                "type": "int"
                            }
                        }
                    }
                }
            }
