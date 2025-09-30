import os

from config.model.ModelBelongTypeStr import ModelBelongTypeStr
from config.model.ModelTypeLabelStr import ModelTypeLabelStr
from config.model.ModelTypeNameStr import ModelTypeNameStr


class ModelNameAndModelTypeParser:
    @classmethod
    def read_model_type_list(cls):
        model_type_list = ModelTypeLabelStr.get_all()

        out_list = []
        for i, type in enumerate(model_type_list):
            out_list.append({
                "id": i,
                "name": ModelTypeNameStr.mapping()[type],
                "value": type
            })

        return out_list

    @classmethod
    def read_model_name_dict(cls):

        model_type_list = ModelTypeLabelStr.get_all()

        model_name_dict = {}
        for model_type in model_type_list:

            model_name_list = []
            for i, model_name in enumerate(ModelBelongTypeStr.get_model_name_list_on_model_type(model_type)):

                model_name_list.append({
                    "id": i,
                    "name": model_name,
                    "value": model_name
                })

            model_name_dict[model_type] = model_name_list

        return model_name_dict
