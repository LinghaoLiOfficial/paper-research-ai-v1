from config.model.ModelTypeLabelStr import ModelTypeLabelStr
from config.name.data_analysis.AnalysisModelEnName import AnalysisModelEnName


# 模型类型英文名对应模型英文名
class ModelBelongTypeStr:

    @classmethod
    def get_model_type_dict(cls):
        return {
            ModelTypeLabelStr.MACHINE_LEARNING_SUPERVISED_MODEL: [
                AnalysisModelEnName.LightGBM,
                AnalysisModelEnName.SVM,
            ],
            ModelTypeLabelStr.DEEP_LEARNING_SUPERVISED_MODEL: [
                AnalysisModelEnName.LSTM,
                AnalysisModelEnName.Transformer,
                AnalysisModelEnName.AttentionVAE,
                AnalysisModelEnName.BayesianNN
            ],
            ModelTypeLabelStr.DEEP_LEARNING_SELF_SUPERVISED_MODEL: [
                AnalysisModelEnName.AutoEncoder,
                AnalysisModelEnName.RAE,
                AnalysisModelEnName.VAE,
                AnalysisModelEnName.RVAE,
                AnalysisModelEnName.AttentionVAE
            ],
            ModelTypeLabelStr.DEEP_LEARNING_CONTRASTIVE_SELF_SUPERVISED_MODEL: [
                AnalysisModelEnName.ContrastiveBRAE
            ]
        }

    @classmethod
    def get_model_type_on_model_name(cls, model_name):
        model_type_dict = cls.get_model_type_dict()
        for model_type, model_name_list in model_type_dict.items():
            if model_name in model_name_list:
                return model_type

        return None

    @classmethod
    def get_model_name_list_on_model_type(cls, model_type):
        return cls.get_model_type_dict()[model_type]
