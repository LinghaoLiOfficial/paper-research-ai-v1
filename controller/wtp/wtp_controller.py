from flask import Blueprint

from service.wtp.WTPService import WTPService
from entity.common.Req import Req

# 实例化/wtp的Blueprint
wtp_bp = Blueprint(
    name="wtp",
    import_name=__name__,
    url_prefix="/wtp"
)


# 【GET】【/getHistoryParams】获取历史记录超参数
@wtp_bp.get("/getHistoryParams")
def get_history_params_api():
    nick_name = Req.receive_get_param("nickName")
    run_name = Req.receive_get_param("runName")

    return WTPService.get_history_params(
        nick_name=nick_name,
        run_name=run_name
    )


# 【GET】【/getTrainingHistory】获取历史记录
@wtp_bp.get("/getTrainingHistory")
def get_training_history_api():
    return WTPService.get_training_history()


# 【GET】【/checkIfTrainModel】检查服务器是否正在训练模型
@wtp_bp.get("/checkIfTrainModel")
def check_if_train_model_api():
    return WTPService.check_if_train_model()


# 【POST】【/startTrainingModel】开始训练模型
@wtp_bp.post("/startTrainingModel")
def start_training_model_api():
    nick_name = Req.receive_post_param("nickName")
    run_name = Req.receive_post_param("runName")
    current_config = Req.receive_post_param("currentConfig")

    return WTPService.start_training_model(
        nick_name=nick_name,
        run_name=run_name,
        current_config=current_config
    )


#【GET】【/getBlankManualModelInfoConfig】获取空白模型信息配置模板
@wtp_bp.get("/getBlankManualModelInfoConfig")
def get_blank_manual_model_info_config_api():
    model_name = Req.receive_get_param("modelName")
    hyper_params_method = Req.receive_get_param("hyperParamsMethod")

    return WTPService.get_blank_manual_model_info_config(
        model_name=model_name,
        hyper_params_method=hyper_params_method
    )


#【GET】【/getBlankManualConfig】获取空白配置模板
@wtp_bp.get("/getBlankManualConfig")
def get_blank_manual_config_api():
    return WTPService.get_blank_manual_config()


# 【GET】【/getModelInfo】获取模型信息
@wtp_bp.get("/getModelInfo")
def get_model_info_api():
    model_name = Req.receive_get_param("modelName")

    return WTPService.get_model_info(model_name=model_name)


# 【GET】【/getModelImage】获取模型图像
@wtp_bp.get("/getModelImage")
def get_model_image_api():
    model_name = Req.receive_get_param("modelName")

    return WTPService.get_model_image(model_name=model_name)


# 【GET】【/getModelList】获取模型列表
@wtp_bp.get("/getModelList")
def get_model_list_api():
    return WTPService.get_model_list()


# 【GET】【/getTimeScaleValueList】获取时序聚合方法列表
@wtp_bp.get("/getTimeScaleValueList")
def get_time_scale_value_list_api():
    return WTPService.get_time_scale_value_list()


# 【GET】【/getSampleDataMethodList】获取数据采样方法列表
@wtp_bp.get("/getSampleDataMethodList")
def get_sample_data_method_list_api():
    return WTPService.get_sample_data_method_list()


# 【GET】【/getNormalizeDataMethodList】获取数据标准化方法列表
@wtp_bp.get("/getNormalizeDataMethodList")
def get_normalize_data_method_list_api():
    return WTPService.get_normalize_data_method_list()


# 【GET】【/getTargetValueMethodList】获取处理目标值方法列表
@wtp_bp.get("/getTargetValueMethodList")
def get_target_value_method_list_api():
    return WTPService.get_target_value_method_list()


# 【GET】【/getNullValueMethodList】获取处理缺失值方法列表
@wtp_bp.get("/getNullValueMethodList")
def get_null_value_method_list_api():
    return WTPService.get_null_value_method_list()


# 【GET】【/getHyperParamConfigureList】获取超参数配置方法列表
@wtp_bp.get("/getHyperParamConfigureList")
def get_hyper_param_configure_list_api():
    return WTPService.get_hyper_param_configure_list()


# 【GET】【/getDataList】获取数据源列表
@wtp_bp.get("/getDataList")
def get_data_list_api():
    return WTPService.get_data_list()


# 【GET】【/getHyperParamList】获取超参数列表
@wtp_bp.get("/getHyperParamList")
def get_hyper_param_list_api():
    return WTPService.get_hyper_param_list()


