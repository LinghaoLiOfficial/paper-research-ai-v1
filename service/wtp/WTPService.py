import os

from buffer.ModelTrainingBuffer import ModelTrainingBuffer
from config.model.ModelBelongTypeStr import ModelBelongTypeStr
from mapper.wtp.WTPMapper import WTPMapper
from utils.common.HalfAutoModelInfoConfigTemplateParser import HalfAutoModelInfoConfigTemplateParser
from utils.common.LocalDataFileReader import LocalDataFileReader
from utils.data_analysis.ModelNameAndModelTypeParser import ModelNameAndModelTypeParser
from utils.common.ManualConfigTemplateParser import ManualConfigTemplateParser
from utils.common.ManualModelInfoConfigTemplateParser import ManualModelInfoConfigTemplateParser
from utils.data_analysis.ModelInfoParser import ModelInfoParser
from utils.data_analysis.NormalizeDataMethodParser import NormalizeDataMethodParser
from utils.data_analysis.NullValueMethodParser import NullValueMethodParser
from utils.data_analysis.ParamMethodParser import ParamMethodParser
from utils.data_analysis.ParamsParser import ParamsParser
from utils.data_analysis.SampleDataMethodParser import SampleDataMethodParser
from utils.data_analysis.TargetValueMethodParser import TargetValueMethodParser
from utils.data_analysis.TimeScaleValueParser import TimeScaleValueParser
from utils.data_analysis.TrainingHistorySaver import TrainingHistorySaver
from entity.common.Resp import Resp


class WTPService:

    @classmethod
    def get_history_params(cls, nick_name, run_name):
        mysql_result = WTPMapper.select_best_training_history_where_history_owner_and_history_name(
            history_owner=nick_name,
            history_name=run_name
        )
        if not mysql_result.check:
            return Resp.build_db_error()

        history_params = mysql_result.get_data_on_results()[0].history_params

        return Resp.build_success(data={
            "historyParams": history_params
        })

    @classmethod
    def get_training_history(cls):
        training_history = TrainingHistorySaver.load()

        return Resp.build_success(data={
            "trainingHistory": training_history
        })

    @classmethod
    def check_if_train_model(cls):
        train_model_status = ModelTrainingBuffer.check_running_status()
        if not train_model_status:
            return Resp.build_error()

        return Resp.build_success()

    @classmethod
    def start_training_model(cls, nick_name, run_name, current_config):
        ModelTrainingBuffer.start_training(
            nick_name=nick_name,
            run_name=run_name,
            current_config=current_config
        )

        return Resp.build_success()

    @classmethod
    def get_blank_manual_model_info_config(cls, model_name, hyper_params_method):
        model_type = ModelBelongTypeStr.get_model_type_on_model_name(model_name)

        if hyper_params_method == "all_manual":
            blank_model_info_config = ManualModelInfoConfigTemplateParser.generate_blank_template(
                model_name=model_name,
                model_type=model_type
            )
        elif hyper_params_method == "half_auto":
            blank_model_info_config = HalfAutoModelInfoConfigTemplateParser.generate_blank_template(
                model_name=model_name,
                model_type=model_type
            )
        else:
            blank_model_info_config = ManualModelInfoConfigTemplateParser.generate_blank_template(
                model_name=model_name,
                model_type=model_type
            )

        return Resp.build_success(data={
            "blankModelInfoConfig": blank_model_info_config
        })

    @classmethod
    def get_blank_manual_config(cls):
        blank_config = ManualConfigTemplateParser.generate_blank_template()

        return Resp.build_success(data={
            "blankConfig": blank_config
        })

    @classmethod
    def get_model_info(cls, model_name):
        model_type = ModelBelongTypeStr.get_model_type_on_model_name(model_name)
        model_info = ModelInfoParser.get(model_name, model_type)

        return Resp.build_success(data={
            "modelInfo": model_info
        })

    @classmethod
    def get_model_image(cls, model_name):
        # TODO url需要改
        # model_image_url = f"{os.getenv('URL')}/file/static-image-model-{model_name}.png"
        model_image_url = f"{os.getenv('URL')}/file/static-image-test-{model_name}.svg"

        return Resp.build_success(data={
            "modelImageUrl": model_image_url
        })

    @classmethod
    def get_model_list(cls):
        model_type_list = ModelNameAndModelTypeParser.read_model_type_list()
        model_name_dict = ModelNameAndModelTypeParser.read_model_name_dict()

        return Resp.build_success(data={
            "modelTypeList": model_type_list,
            "modelNameDict": model_name_dict
        })

    @classmethod
    def get_time_scale_value_list(cls):
        time_scale_value_list = TimeScaleValueParser.get()

        return Resp.build_success(data={
            "timeScaleValueList": time_scale_value_list
        })

    @classmethod
    def get_sample_data_method_list(cls):
        sample_data_method_list = SampleDataMethodParser.get()

        return Resp.build_success(data={
            "sampleDataMethodList": sample_data_method_list
        })

    @classmethod
    def get_normalize_data_method_list(cls):
        normalize_data_method_list = NormalizeDataMethodParser.get()

        return Resp.build_success(data={
            "normalizeDataMethodList": normalize_data_method_list
        })

    @classmethod
    def get_target_value_method_list(cls):
        target_value_method_list = TargetValueMethodParser.get()

        return Resp.build_success(data={
            "targetValueMethodList": target_value_method_list
        })

    @classmethod
    def get_null_value_method_list(cls):
        null_value_method_list = NullValueMethodParser.get()

        return Resp.build_success(data={
            "nullValueMethodList": null_value_method_list
        })

    @classmethod
    def get_hyper_param_configure_list(cls):
        param_method_list = ParamMethodParser.get()

        return Resp.build_success(data={
            "paramMethodList": param_method_list
        })

    @classmethod
    def get_hyper_param_list(cls):
        hyper_params_list = ParamsParser.get()

        return Resp.build_success(data={
            "hyperParamsList": hyper_params_list
        })

    @classmethod
    def get_data_list(cls):
        folder_name_list = LocalDataFileReader.read_folder_name_list()
        data_file_name_dict = LocalDataFileReader.read_file_name_dict()

        return Resp.build_success(data={
            "folderNameList": folder_name_list,
            "dataFileNameDict": data_file_name_dict
        })



