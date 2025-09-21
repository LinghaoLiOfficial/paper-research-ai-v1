from config.model.ModelTypeLabelStr import ModelTypeLabelStr
from config.name.data_analysis.AnalysisModelEnName import AnalysisModelEnName


class ManualConfigTemplateParser:
    @classmethod
    def generate_blank_template(cls):
        BLANK_DEFAULT_VALUE = None

        return {
            "dataSetting": {
                "configureHyperParams": {
                    "method": "half_auto",
                    "data": {
                        "epochs": 20,
                        "lr": 0.0001,
                        "train_ratio": 0.8,
                        "batch_size": 64,
                        "predict_range": 1,
                        "random_seed": 56,
                        "timestep": 40
                    }
                },
                "dataImport": {
                    "chooseDataset": "su_you",
                    "chooseDataTable": "rval_history_i01001_df.csv"
                },
                "dataProcess": {
                    "timeScaleValue": {
                        "status": True,
                        "method": "time_scale_360"
                    },
                    "dealNullValue": {
                        "status": True,
                        "method": "fill_null",
                        "data": 0
                    },
                    "dealTargetValue": {
                        "status": True,
                        "method": "all_binarize",
                    },
                    "normalizeData": {
                        "status": True,
                        "method": "z_score_plus_min_max",
                    },
                    "sampleData": {
                        "status": False,
                        "method": BLANK_DEFAULT_VALUE,
                    }
                }
            },

            "modelSetting": {
                "modelType": ModelTypeLabelStr.DEEP_LEARNING_CONTRASTIVE_SELF_SUPERVISED_MODEL,
                "modelName": AnalysisModelEnName.ContrastiveBRAE,
                "modelInfo": BLANK_DEFAULT_VALUE
            }
        }
