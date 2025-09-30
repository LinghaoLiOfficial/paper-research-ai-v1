from flask import Blueprint

from entity.common.Req import Req
from service.DataAnalysisService import DataAnalysisService

# 实例化Blueprint
da_bp = Blueprint(
    name="da",
    import_name=__name__,
    url_prefix="/da"
)


# 获取默认配置信息
@da_bp.get("/getDefaultDataAnalysisConfig")
def get_default_data_analysis_config_api():

    return DataAnalysisService.get_default_data_analysis_config()


# 获取数据库选项
@da_bp.get("/getAnalysisDataSetOptions")
def get_analysis_data_set_options_api():
    task_type = Req.receive_get_param("taskType")
    data_type = Req.receive_get_param("dataType")

    return DataAnalysisService.get_analysis_data_set_options(
        task_type=task_type,
        data_type=data_type
    )


# 获取数据表选项
@da_bp.get("/getAnalysisDataTableOptions")
def get_analysis_data_table_options_api():
    task_type = Req.receive_get_param("taskType")
    data_type = Req.receive_get_param("dataType")
    data_set = Req.receive_get_param("dataSet")

    return DataAnalysisService.get_analysis_data_table_options(
        task_type=task_type,
        data_type=data_type,
        data_set=data_set
    )


# 获取时序聚合选项
@da_bp.get("/getAnalysisTimeScaleOptions")
def get_analysis_time_scale_options_api():

    return DataAnalysisService.get_analysis_time_scale_options()


# 获取处理缺失值选项
@da_bp.get("/getAnalysisNullValueOptions")
def get_analysis_null_value_options_api():

    return DataAnalysisService.get_analysis_null_value_options()


# 获取处理目标值选项
@da_bp.get("/getAnalysisTargetValueOptions")
def get_analysis_target_value_options_api():
    task_type = Req.receive_get_param("taskType")

    return DataAnalysisService.get_analysis_target_value_options(
        task_type=task_type
    )


# 获取数据标准化选项
@da_bp.get("/getAnalysisNormalizeValueOptions")
def get_analysis_normalize_value_options_api():

    return DataAnalysisService.get_analysis_normalize_value_options()


# 获取数据采样选项
@da_bp.get("/getAnalysisSampleValueOptions")
def get_analysis_sample_value_options_api():

    return DataAnalysisService.get_analysis_sample_value_options()


# 获取模型类型选项
@da_bp.get("/getAnalysisModelTypeOptions")
def get_analysis_model_type_options_api():

    return DataAnalysisService.get_analysis_model_type_options()


# 获取模型选项
@da_bp.get("/getAnalysisModelOptions")
def get_analysis_model_options_api():
    model_type = Req.receive_get_param("modelType")

    return DataAnalysisService.get_analysis_model_options(model_type)


# 获取模型参数选项
@da_bp.get("/getAnalysisParams")
def get_analysis_params_api():
    return DataAnalysisService.get_analysis_params()


# 获取模型信息
@da_bp.get("/getAnalysisModelInfo")
def get_analysis_model_info_api():
    model = Req.receive_get_param("model")

    return DataAnalysisService.get_analysis_model_info(model)


# 获取数据表的dataframe
@da_bp.get("/getDataFrame")
def get_dataframe_api():
    task_type = Req.receive_get_param("taskType")
    data_type = Req.receive_get_param("dataType")
    data_set = Req.receive_get_param("dataSet")
    data_table = Req.receive_get_param("dataTable")

    return DataAnalysisService.get_dataframe(
        task_type=task_type,
        data_type=data_type,
        data_set=data_set,
        data_table=data_table
    )


# 开始训练模型
@da_bp.post("/startDataAnalysisTraining")
def start_data_analysis_training_api():
    data_analysis_config = Req.receive_post_param("dataAnalysisConfig")
    task_name = Req.receive_post_param("taskName")
    token = Req.receive_header_token()

    return DataAnalysisService.start_data_analysis_training(
        data_analysis_config=data_analysis_config,
        task_name=task_name,
        token=token
    )


# 获取模型训练历史记录
@da_bp.get("/getTrainingHistory")
def get_training_history_api():
    task_type = Req.receive_get_param("taskType")
    task_level = Req.receive_get_param("taskLevel")
    token = Req.receive_header_token()

    return DataAnalysisService.get_training_history(
        task_type=task_type,
        task_level=task_level,
        token=token
    )


# 获取历史所有模型训练历史记录
@da_bp.get("/getAllTrainingHistory")
def get_all_training_history_api():
    task_type = Req.receive_get_param("taskType")
    task_id = Req.receive_get_param("taskId")
    study_id = Req.receive_get_param("studyId")
    run_id = Req.receive_get_param("runId")
    token = Req.receive_header_token()

    return DataAnalysisService.get_all_training_history(
        task_type=task_type,
        task_id=task_id,
        study_id=study_id,
        run_id=run_id,
        token=token
    )


# 获取当前模型训练历史记录的超参数
@da_bp.get("/getHistoryParams")
def get_history_params_api():
    task_type = Req.receive_get_param("taskType")
    task_id = Req.receive_get_param("taskId")
    study_id = Req.receive_get_param("studyId")
    run_id = Req.receive_get_param("runId")
    token = Req.receive_header_token()

    return DataAnalysisService.get_history_params(
        task_type=task_type,
        task_id=task_id,
        study_id=study_id,
        run_id=run_id,
        token=token
    )


# 获取当前模型训练历史记录的超参数
@da_bp.get("/getTaskNames")
def get_task_names_api():
    task_type = Req.receive_get_param("taskType")
    token = Req.receive_header_token()

    return DataAnalysisService.get_task_names(
        task_type=task_type,
        token=token
    )


# 获取当前模型训练记录的所有图表
@da_bp.get("/getAllVisualizeImages")
def get_all_visualize_images_api():
    task_id = Req.receive_get_param("taskId")
    run_id = Req.receive_get_param("runId")
    token = Req.receive_header_token()

    return DataAnalysisService.get_all_visualize_images(
        task_id=task_id,
        run_id=run_id,
        token=token
    )


# 获取当前数据表的默认目标变量列的值
@da_bp.get("/getDefaultTargetColumn")
def get_default_target_column_api():
    data_type = Req.receive_get_param("dataType")
    data_set = Req.receive_get_param("dataSet")
    token = Req.receive_header_token()

    return DataAnalysisService.get_default_target_column(
        data_type=data_type,
        data_set=data_set,
        token=token
    )


# 清除无效的数据分析文件
@da_bp.post("/activateClearFiles")
def activate_clear_files_api():
    task_type = Req.receive_post_param("taskType")
    token = Req.receive_header_token()

    return DataAnalysisService.activate_clear_files(
        task_type=task_type,
        token=token
    )


# 获取超参数优化选项列表
@da_bp.get("/getAnalysisHyperParamsOptions")
def get_analysis_hyper_params_options_api():
    token = Req.receive_header_token()

    return DataAnalysisService.get_analysis_hyper_params_options(
        token=token
    )


# 获取任务模板选项列表
@da_bp.get("/getAnalysisPatternOptions")
def get_analysis_pattern_options_api():
    token = Req.receive_header_token()

    return DataAnalysisService.get_analysis_pattern_options(
        token=token
    )


# 删除当前任务的所有记录
@da_bp.post("/deleteHistory")
def delete_history_api():
    task_type = Req.receive_post_param("taskType")
    task_id = Req.receive_post_param("taskId")
    run_id = Req.receive_post_param("runId")
    token = Req.receive_header_token()

    return DataAnalysisService.delete_history(
        task_type=task_type,
        task_id=task_id,
        run_id=run_id,
        token=token
    )