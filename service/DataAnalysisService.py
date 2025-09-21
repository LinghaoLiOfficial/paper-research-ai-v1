import os
from datetime import datetime
import pandas as pd
import shutil

from buffer.ModelTrainingBuffer import ModelTrainingBuffer
from config.name.data_analysis.AnalysisDataTypeEnName import AnalysisDataTypeEnName
from config.name.data_analysis.AnalysisHyperParamsOptimizeEnName import AnalysisHyperParamsOptimizeEnName
from config.name.data_analysis.AnalysisHyperParamsOptimizeZhName import AnalysisHyperParamsOptimizeZhName
from config.name.data_analysis.AnalysisModelEnName import AnalysisModelEnName
from config.name.data_analysis.AnalysisModelTypeEnName import AnalysisModelTypeEnName
from config.name.data_analysis.AnalysisModelTypeZhName import AnalysisModelTypeZhName
from config.name.data_analysis.AnalysisNormalizeValueEnName import AnalysisNormalizeValueEnName
from config.name.data_analysis.AnalysisNormalizeValueZhName import AnalysisNormalizeValueZhName
from config.name.data_analysis.AnalysisNullValueEnName import AnalysisNullValueEnName
from config.name.data_analysis.AnalysisNullValueZhName import AnalysisNullValueZhName
from config.name.data_analysis.AnalysisParamsDefault import AnalysisParamsDefault
from config.name.data_analysis.AnalysisParamsEnName import AnalysisParamsEnName
from config.name.data_analysis.AnalysisParamsZhName import AnalysisParamsZhName
from config.name.data_analysis.AnalysisSampleValueEnName import AnalysisSampleValueEnName
from config.name.data_analysis.AnalysisSampleValueZhName import AnalysisSampleValueZhName
from config.name.data_analysis.AnalysisTargetValueEnName import AnalysisTargetValueEnName
from config.name.data_analysis.AnalysisTargetValueZhName import AnalysisTargetValueZhName
from config.name.data_analysis.AnalysisTaskTypeEnName import AnalysisTaskTypeEnName
from config.name.data_analysis.AnalysisTimeScaleEnName import AnalysisTimeScaleEnName
from config.name.data_analysis.AnalysisTimeScaleZhName import AnalysisTimeScaleZhName
from config.name.data_analysis.model_info.BiScaleWaveNetModelInfo import BiScaleWaveNetModelInfo
from config.name.data_analysis.module.ContrastiveBRAEName import ContrastiveBRAEName
from config.name.data_analysis.data_table.TimeSeriesDataSetInit import TimeSeriesDataSetInit
from config.name.data_analysis.model_info.AttentionLSTMModelInfo import AttentionLSTMModelInfo
from config.name.data_analysis.model_info.AttentionVAEModelInfo import AttentionVAEModelInfo
from config.name.data_analysis.model_info.AutoEncoderModelInfo import AutoEncoderModelInfo
from config.name.data_analysis.model_info.BayesianNNModelInfo import BayesianNNModelInfo
from config.name.data_analysis.model_info.BayesianRegressionModelInfo import BayesianRegressionModelInfo
from config.name.data_analysis.model_info.ContrastiveBRAEModelInfo import ContrastiveBRAEModelInfo
from config.name.data_analysis.model_info.LSTMModelInfo import LSTMModelInfo
from config.name.data_analysis.model_info.LightGBMModelInfo import LightGBMModelInfo
from config.name.data_analysis.model_info.RAEModelInfo import RAEModelInfo
from config.name.data_analysis.model_info.RVAEModelInfo import RVAEModelInfo
from config.name.data_analysis.model_info.SVMModelInfo import SVMModelInfo
from config.name.data_analysis.model_info.TransformerModelInfo import TransformerModelInfo
from config.name.data_analysis.model_info.VAEModelInfo import VAEModelInfo
from config.name.data_analysis.model_info.WaveNetModelInfo import WaveNetModelInfo
from entity.common.Resp import Resp
from mapper.DataAnalysisMapper import DataAnalysisMapper
from pattern.ModelPatternFactory import ModelPatternFactory
from utils.common.JWTParser import JWTParser
from utils.common.RandomStrGenerator import RandomStrGenerator
from utils.common.TimeParser import TimeParser
from utils.data_analysis.TrainingHistorySaver import TrainingHistorySaver


class DataAnalysisService:

    ANALYSIS_DATA_PATH = "./storage/{}/data_analysis/{}"

    @classmethod
    def delete_history(cls, task_type, task_id, run_id, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        if task_type == AnalysisTaskTypeEnName.CLASSIFICATION:
            mysql_result = DataAnalysisMapper.delete_classification_training_history({
                "task_id": task_id
            })
            mysql_result1 = DataAnalysisMapper.delete_best_classification_training_history({
                "task_id": task_id
            })
        else:
            mysql_result = DataAnalysisMapper.delete_regression_training_history({
                "task_id": task_id
            })
            mysql_result1 = DataAnalysisMapper.delete_best_regression_training_history({
                "task_id": task_id
            })

        return Resp.build_success()

    @classmethod
    def get_analysis_pattern_options(cls, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        analysis_task_pattern_options = [{
            "id": i,
            "label": pattern_name,
            "name": pattern_name
        } for i, pattern_name in enumerate(ModelPatternFactory.get_all_model_pattern_name())]

        return Resp.build_success(data={
            "analysisHyperParamsOptions": analysis_task_pattern_options
        })

    @classmethod
    def activate_clear_files(cls, task_type, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        if task_type == AnalysisTaskTypeEnName.CLASSIFICATION:
            mysql_result = DataAnalysisMapper.select_best_classification_training_history_where_user_id({
                "history_owner": user_id
            })
        else:
            mysql_result = DataAnalysisMapper.select_best_regression_training_history_where_user_id({
                "history_owner": user_id
            })

        valid_task_name_list = list(set([record['task_id'] for record in mysql_result.get_data_on_results()]))

        root_path = cls.ANALYSIS_DATA_PATH.format(user_id, "").rstrip("/{}")
        for folder_name in os.listdir(root_path):
            if folder_name not in valid_task_name_list:
                try:
                    shutil.rmtree(f"{root_path}/{folder_name}")
                except Exception as e:
                    print(e)

        return Resp.build_success()

    @classmethod
    def get_analysis_hyper_params_options(cls, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        analysis_hyper_params_options = [
            {
                "id": 0,
                "label": AnalysisHyperParamsOptimizeZhName.BAYESIAN_OPTIMIZE,
                "name": AnalysisHyperParamsOptimizeEnName.BAYESIAN_OPTIMIZE
            },
        ]

        return Resp.build_success(data={
            "analysisHyperParamsOptions": analysis_hyper_params_options
        })

    @classmethod
    def get_default_target_column(cls, data_type, data_set, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        target_name_value = {
            AnalysisDataTypeEnName.TIME_SERIES: TimeSeriesDataSetInit.init.get(data_set),
            AnalysisDataTypeEnName.COMMON: None,
            AnalysisDataTypeEnName.GRAPH: None
        }.get(data_type)

        target_name = {
            "target_name": target_name_value
        }

        return Resp.build_success(data={
            "targetName": target_name
        })

    @classmethod
    def get_all_visualize_images(cls, task_id, run_id, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        image_base_path = f"{cls.ANALYSIS_DATA_PATH.format(user_id, task_id)}/{run_id}"

        # 获取所有图片文件URL
        images = []
        for image_name in os.listdir(image_base_path):
            if image_name.endswith(".svg"):
                image_url = f"{os.getenv('URL')}/file{image_base_path.lstrip('.')}/{image_name}"
                images.append({
                    "name": image_name.split(".")[0],
                    "url": image_url
                })

        return Resp.build_success(data={
            "visualizeImages": images
        })

    @classmethod
    def get_task_names(cls, task_type, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        if task_type == AnalysisTaskTypeEnName.CLASSIFICATION:
            mysql_result = DataAnalysisMapper.select_best_classification_training_history_where_user_id({
                "history_owner": user_id
            })
        else:
            mysql_result = DataAnalysisMapper.select_best_regression_training_history_where_user_id({
                "history_owner": user_id
            })

        # 去重任务名
        task_name_list = list(set([record['task_name'] for record in mysql_result.get_data_on_results()]))

        return Resp.build_success(data={
            "taskNames": task_name_list
        })

    @classmethod
    def get_history_params(cls, task_type, task_id, study_id, run_id, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        if task_type == AnalysisTaskTypeEnName.CLASSIFICATION:
            mysql_result = DataAnalysisMapper.select_best_classification_training_history_where_task_id_and_study_id_and_run_id({
                "task_id": task_id,
                "study_id": study_id,
                "run_id": run_id
            })
        else:
            mysql_result = DataAnalysisMapper.select_best_regression_training_history_where_task_id_and_study_id_and_run_id({
                "task_id": task_id,
                "study_id": study_id,
                "run_id": run_id
            })

        history_params = mysql_result.get_data_on_results()[0]['history_params']

        # 修正json格式
        history_params = history_params.replace("'", '"').replace("True", "true").replace("False", "false")

        return Resp.build_success(data={
            "historyParams": history_params
        })

    @classmethod
    def get_all_training_history(cls, task_type, task_id, study_id, run_id, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        if task_type == AnalysisTaskTypeEnName.CLASSIFICATION:
            mysql_result = DataAnalysisMapper.select_classification_training_history({
                "task_id": task_id,
                "study_id": study_id,
                "run_id": run_id
            })
        else:
            mysql_result = DataAnalysisMapper.select_regression_training_history({
                "task_id": task_id,
                "study_id": study_id,
                "run_id": run_id
            })

        training_history_list = mysql_result.get_data_on_results()

        head = []
        body = []
        if len(training_history_list) > 0:
            head = [x for x in training_history_list[0].keys() if x not in ["history_id", "history_owner", "history_params"]]

            for history in training_history_list:
                history['history_timestamp'] = TimeParser.convert_time_format(history['history_timestamp'])
                history['history_epoch'] = str(history['history_epoch'])

                body.append([history[x] for x in history.keys() if x not in ["history_id", "history_owner", "history_params"]])

            body = sorted(body, key=lambda v: datetime.strptime(v[17], '%Y-%m-%d %H:%M:%S'), reverse=True)

        training_history = {
            "head": head,
            "body": body
        }

        return Resp.build_success(data={
            "trainingHistory": training_history
        })

    @classmethod
    def get_training_history(cls, task_type, task_level, token):

        # 解析token
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        if task_type == AnalysisTaskTypeEnName.CLASSIFICATION:
            mysql_result = DataAnalysisMapper.select_best_classification_training_history_where_user_id({
                "history_owner": user_id
            })
        else:
            mysql_result = DataAnalysisMapper.select_best_regression_training_history_where_user_id({
                "history_owner": user_id
            })

        best_training_history_list = mysql_result.get_data_on_results()

        compare_col_name = 'test_f1'

        if task_level == 'study':

            groups = {}
            for item in best_training_history_list:
                key = (item['history_owner'], item['task_id'], item['task_name'], item['study_id'])
                if key not in groups:
                    groups[key] = []
                groups[key].append(item)

            combined_best_training_history_list = []
            for group in groups.values():
                if not group:
                    continue  # 防止空分组，虽然正常情况下不会出现
                max_d = max(item[compare_col_name] for item in group)
                max_items = [item for item in group if item[compare_col_name] == max_d]
                combined_best_training_history_list.extend(max_items)

            best_training_history_list = combined_best_training_history_list

        head = []
        body = []
        if len(best_training_history_list) > 0:
            head = [x for x in best_training_history_list[0].keys() if x not in ["history_id", "history_owner", "history_params"]]

            for history in best_training_history_list:
                history['history_timestamp'] = TimeParser.convert_time_format(history['history_timestamp'])
                history['history_epoch'] = str(history['history_epoch'])

                body.append([history[x] for x in history.keys() if x not in ["history_id", "history_owner", "history_params"]])

            body = sorted(body, key=lambda v: datetime.strptime(v[11], '%Y-%m-%d %H:%M:%S'), reverse=True)

        training_history = {
            "head": head,
            "body": body
        }

        return Resp.build_success(data={
            "trainingHistory": training_history
        })

    @classmethod
    def get_analysis_data_table_options(cls, task_type, data_type, data_set):

        data_path = f"./data/{task_type}/{data_type}/{data_set}"
        analysis_data_table_options = [
            {
                "id": index,
                "label": file_name,
                "name": file_name
            } for index, file_name in enumerate(os.listdir(data_path))
        ]

        return Resp.build_success(data={
            "analysisDataTableOptions": analysis_data_table_options
        })

    @classmethod
    def get_dataframe(cls, task_type, data_type, data_set, data_table):

        dataframe_path = f"./data/{task_type}/{data_type}/{data_set}/{data_table}"
        df = pd.read_csv(dataframe_path)

        # 只显示前50条数据
        show_df = df.iloc[:50, :]

        head = show_df.columns.tolist()
        body = show_df.values.tolist()

        # 获取数据的相关信息
        row_num = df.shape[0]
        col_num = df.shape[1]
        repeated_row_num = int(df.duplicated().sum())
        null_col_list = ", ".join(df.columns[df.isnull().any()].tolist())

        analysis_dataframe = {
            'head': head,
            'body': body,
            'info': {
                'rowNum': row_num,
                'colNum': col_num,
                'repeatedRowNum': repeated_row_num,
                'nullColList': null_col_list
            }
        }

        return Resp.build_success(data={
            "analysisDataframe": analysis_dataframe
        })

    @classmethod
    def start_data_analysis_training(cls, data_analysis_config, task_name, token):
        jwt_parser_result = JWTParser.decode_user_id(
            token=token
        )
        if not jwt_parser_result.check:
            return Resp.build_jwt_error(jwt_parser_result)

        user_id = jwt_parser_result.get_data_on_results()

        # 检查用户权限
        mysql_result = DataAnalysisMapper.select_user({
            "user_id": user_id
        })

        if mysql_result.get_data_on_results()[0]['user_auth'] != 'admin':
            return Resp.build_error(
                code=50001,
                message="抱歉，您的权限等级不足"
            )

        # 生成唯一的task_id
        while True:
            task_id = RandomStrGenerator.generate_5_random_str()
            base_path = cls.ANALYSIS_DATA_PATH.format(user_id, task_id)
            if not os.path.exists(base_path):
                break

        for root, dirs, files in os.walk(cls.ANALYSIS_DATA_PATH.rstrip("/{}").format(user_id), topdown=False):
            # 遍历当前文件夹中的子文件夹
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)  # 子文件夹的完整路径
                # 如果子文件夹为空，则删除
                if not os.listdir(dir_path):  # os.listdir() 返回文件夹中的内容列表
                    os.rmdir(dir_path)  # 删除空文件夹

        # 若不存在则创建文件夹
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        data_analysis_config['task_name'] = task_name
        data_analysis_config['user_id'] = user_id
        data_analysis_config['task_id'] = task_id
        data_analysis_config['base_path'] = base_path

        model_training_buffer_result = ModelTrainingBuffer.start_running(
            config=data_analysis_config
        )

        return Resp.build_success(
            code=model_training_buffer_result.code,
            message=model_training_buffer_result.message
        )

    @classmethod
    def get_analysis_model_info(cls, model):
        model_info = {
            AnalysisModelEnName.BayesianRegression: BayesianRegressionModelInfo.to_list(),
            AnalysisModelEnName.LSTM: LSTMModelInfo.to_list(),
            AnalysisModelEnName.Transformer: TransformerModelInfo.to_list(),
            AnalysisModelEnName.AttentionLSTM: AttentionLSTMModelInfo.to_list(),
            AnalysisModelEnName.BayesianNN: BayesianNNModelInfo.to_list(),
            AnalysisModelEnName.AutoEncoder: AutoEncoderModelInfo.to_list(),
            AnalysisModelEnName.RAE: RAEModelInfo.to_list(),
            AnalysisModelEnName.VAE: VAEModelInfo.to_list(),
            AnalysisModelEnName.RVAE: RVAEModelInfo.to_list(),
            AnalysisModelEnName.AttentionVAE: AttentionVAEModelInfo.to_list(),
            AnalysisModelEnName.ContrastiveBRAE: ContrastiveBRAEModelInfo.to_list(),
            AnalysisModelEnName.LightGBM: LightGBMModelInfo.to_list(),
            AnalysisModelEnName.SVM: SVMModelInfo.to_list(),
            AnalysisModelEnName.WaveNet: WaveNetModelInfo.to_list(),
            AnalysisModelEnName.BiScaleWaveNet: BiScaleWaveNetModelInfo.to_list()
        }.get(model)

        return Resp.build_success(data={
            "modelInfo": model_info
        })

    @classmethod
    def get_analysis_params(cls):
        analysis_params = [
            {
                "id": 0,
                "label": AnalysisParamsZhName.EPOCHS,
                "name": AnalysisParamsEnName.EPOCHS
            },
            {
                "id": 1,
                "label": AnalysisParamsZhName.LR,
                "name": AnalysisParamsEnName.LR
            },
            {
                "id": 2,
                "label": AnalysisParamsZhName.TRAIN_RATIO,
                "name": AnalysisParamsEnName.TRAIN_RATIO
            },
            {
                "id": 3,
                "label": AnalysisParamsZhName.BATCH_SIZE,
                "name": AnalysisParamsEnName.BATCH_SIZE
            },
            {
                "id": 4,
                "label": AnalysisParamsZhName.PREDICT_RANGE,
                "name": AnalysisParamsEnName.PREDICT_RANGE
            },
            {
                "id": 5,
                "label": AnalysisParamsZhName.RANDOM_SEED,
                "name": AnalysisParamsEnName.RANDOM_SEED
            },
            {
                "id": 6,
                "label": AnalysisParamsZhName.TIMESTEP,
                "name": AnalysisParamsEnName.TIMESTEP
            },
            {
                "id": 7,
                "label": AnalysisParamsZhName.WEIGHT_DECAY,
                "name": AnalysisParamsEnName.WEIGHT_DECAY
            },
        ]

        return Resp.build_success(data={
            "analysisParams": analysis_params
        })

    @classmethod
    def get_analysis_model_options(cls, model_type):
        analysis_model_options = {
            AnalysisModelTypeEnName.DEEP_LEARNING_SUPERVISED_MODEL: [
                {
                    "id": 0,
                    "label": AnalysisModelEnName.LSTM,
                    "name": AnalysisModelEnName.LSTM
                },
                {
                    "id": 1,
                    "label": AnalysisModelEnName.Transformer,
                    "name": AnalysisModelEnName.Transformer
                },
                {
                    "id": 2,
                    "label": AnalysisModelEnName.AttentionLSTM,
                    "name": AnalysisModelEnName.AttentionLSTM
                },
                {
                    "id": 3,
                    "label": AnalysisModelEnName.BayesianNN,
                    "name": AnalysisModelEnName.BayesianNN
                },
                {
                    "id": 4,
                    "label": AnalysisModelEnName.WaveNet,
                    "name": AnalysisModelEnName.WaveNet
                },
                {
                    "id": 5,
                    "label": AnalysisModelEnName.BiScaleWaveNet,
                    "name": AnalysisModelEnName.BiScaleWaveNet
                },
            ],
            AnalysisModelTypeEnName.DEEP_LEARNING_SELF_SUPERVISED_MODEL: [
                {
                    "id": 4,
                    "label": AnalysisModelEnName.AutoEncoder,
                    "name": AnalysisModelEnName.AutoEncoder
                },
                {
                    "id": 5,
                    "label": AnalysisModelEnName.RAE,
                    "name": AnalysisModelEnName.RAE
                },
                {
                    "id": 6,
                    "label": AnalysisModelEnName.VAE,
                    "name": AnalysisModelEnName.VAE
                },
                {
                    "id": 7,
                    "label": AnalysisModelEnName.RVAE,
                    "name": AnalysisModelEnName.RVAE
                },
                {
                    "id": 8,
                    "label": AnalysisModelEnName.AttentionVAE,
                    "name": AnalysisModelEnName.AttentionVAE
                },
            ],
            AnalysisModelTypeEnName.DEEP_LEARNING_CONTRASTIVE_SELF_SUPERVISED_MODEL: [
                {
                    "id": 9,
                    "label": AnalysisModelEnName.ContrastiveBRAE,
                    "name": AnalysisModelEnName.ContrastiveBRAE
                },
            ],
            AnalysisModelTypeEnName.MACHINE_LEARNING_SUPERVISED_MODEL: [
                {
                    "id": 10,
                    "label": AnalysisModelEnName.BayesianRegression,
                    "name": AnalysisModelEnName.BayesianRegression
                },
                {
                    "id": 11,
                    "label": AnalysisModelEnName.LightGBM,
                    "name": AnalysisModelEnName.LightGBM
                },
                {
                    "id": 12,
                    "label": AnalysisModelEnName.SVM,
                    "name": AnalysisModelEnName.SVM
                },
            ]
        }.get(model_type)

        return Resp.build_success(data={
            "analysisModelOptions": analysis_model_options
        })

    @classmethod
    def get_analysis_model_type_options(cls):
        analysis_model_type_options = [
            {
                "id": 0,
                "label": AnalysisModelTypeZhName.DEEP_LEARNING_SUPERVISED_MODEL,
                "name": AnalysisModelTypeEnName.DEEP_LEARNING_SUPERVISED_MODEL
            },
            {
                "id": 1,
                "label": AnalysisModelTypeZhName.DEEP_LEARNING_SELF_SUPERVISED_MODEL,
                "name": AnalysisModelTypeEnName.DEEP_LEARNING_SELF_SUPERVISED_MODEL
            },
            {
                "id": 2,
                "label": AnalysisModelTypeZhName.DEEP_LEARNING_CONTRASTIVE_SELF_SUPERVISED_MODEL,
                "name": AnalysisModelTypeEnName.DEEP_LEARNING_CONTRASTIVE_SELF_SUPERVISED_MODEL
            },
            {
                "id": 3,
                "label": AnalysisModelTypeZhName.MACHINE_LEARNING_SUPERVISED_MODEL,
                "name": AnalysisModelTypeEnName.MACHINE_LEARNING_SUPERVISED_MODEL
            },
        ]

        return Resp.build_success(data={
            "analysisModelTypeOptions": analysis_model_type_options
        })

    @classmethod
    def get_analysis_sample_value_options(cls):
        analysis_sample_value_options = [
            {
                "id": 0,
                "label": AnalysisSampleValueZhName.RANDOM_UNDER,
                "name": AnalysisSampleValueEnName.RANDOM_UNDER
            },
            {
                "id": 1,
                "label": AnalysisSampleValueZhName.SMOTE_OVER,
                "name": AnalysisSampleValueEnName.SMOTE_OVER
            },
        ]

        return Resp.build_success(data={
            "analysisSampleValueOptions": analysis_sample_value_options
        })

    @classmethod
    def get_analysis_normalize_value_options(cls):
        analysis_normalize_value_options = [
            {
                "id": 0,
                "label": AnalysisNormalizeValueZhName.Z_SCORE,
                "name": AnalysisNormalizeValueEnName.Z_SCORE
            },
            {
                "id": 1,
                "label": AnalysisNormalizeValueZhName.MIN_MAX,
                "name": AnalysisNormalizeValueEnName.MIN_MAX
            }
        ]

        return Resp.build_success(data={
            "analysisNormalizeValueOptions": analysis_normalize_value_options
        })

    @classmethod
    def get_analysis_target_value_options(cls, task_type):


        analysis_target_value_options = {
            AnalysisTaskTypeEnName.CLASSIFICATION: [
                {
                    "id": 0,
                    "label": AnalysisTargetValueZhName.MULTI_LABELS,
                    "name": AnalysisTargetValueEnName.MULTI_LABELS
                },
                {
                    "id": 1,
                    "label": AnalysisTargetValueZhName.ALL_BINARIZE,
                    "name": AnalysisTargetValueEnName.ALL_BINARIZE
                },
                {
                    "id": 2,
                    "label": AnalysisTargetValueZhName.SPECIFIC_TARGET,
                    "name": AnalysisTargetValueEnName.SPECIFIC_TARGET,
                    "value": 1
                },
            ],
            AnalysisTaskTypeEnName.REGRESSION: [
                {
                    "id": 0,
                    "label": AnalysisTargetValueZhName.SINGLE_VAR,
                    "name": AnalysisTargetValueEnName.SINGLE_VAR
                },
                {
                    "id": 1,
                    "label": AnalysisTargetValueZhName.MULTI_VAR,
                    "name": AnalysisTargetValueEnName.MULTI_VAR
                }
            ],
        }.get(task_type)

        return Resp.build_success(data={
            "analysisTargetValueOptions": analysis_target_value_options
        })

    @classmethod
    def get_analysis_null_value_options(cls):
        analysis_null_value_options = [
            {
                "id": 0,
                "label": AnalysisNullValueZhName.FILL_NULL_WITH_MEAN,
                "name": AnalysisNullValueEnName.FILL_NULL_WITH_MEAN
            },
            {
                "id": 1,
                "label": AnalysisNullValueZhName.FILL_NULL_WITH_ZERO,
                "name": AnalysisNullValueEnName.FILL_NULL_WITH_ZERO
            },
            {
                "id": 2,
                "label": AnalysisNullValueZhName.DEL_NULL,
                "name": AnalysisNullValueEnName.DEL_NULL
            },
        ]

        return Resp.build_success(data={
            "analysisNullValueOptions": analysis_null_value_options
        })

    @classmethod
    def get_analysis_time_scale_options(cls):
        analysis_time_scale_options = [
            {
                "id": 0,
                "label": AnalysisTimeScaleZhName.SCALE_REDUCE,
                "name": AnalysisTimeScaleEnName.SCALE_REDUCE
            }
        ]

        return Resp.build_success(data={
            "analysisTimeScaleOptions": analysis_time_scale_options
        })

    @classmethod
    def get_analysis_data_set_options(cls, task_type, data_type):

        data_path = f"./data/{task_type}/{data_type}"
        analysis_data_set_options = [
            {
                "id": index,
                "label": file_name,
                "name": file_name
            } for index, file_name in enumerate(os.listdir(data_path))
        ]

        return Resp.build_success(data={
            "analysisDataSetOptions": analysis_data_set_options
        })

    @classmethod
    def get_default_data_analysis_config(cls):
        default_data_analysis_config = {
            "data_setting": {
                "task_pattern": "",
                "task_type": AnalysisTaskTypeEnName.CLASSIFICATION,
                "data_type": AnalysisDataTypeEnName.TIME_SERIES,
                "dataset": {

                },
                "task_params": {
                    AnalysisParamsEnName.EPOCHS: [AnalysisParamsDefault.EPOCHS],
                    AnalysisParamsEnName.LR: [AnalysisParamsDefault.LR, 0.01],
                    AnalysisParamsEnName.TRAIN_RATIO: [AnalysisParamsDefault.TRAIN_RATIO],
                    AnalysisParamsEnName.BATCH_SIZE: [AnalysisParamsDefault.BATCH_SIZE],
                    AnalysisParamsEnName.PREDICT_RANGE: [AnalysisParamsDefault.PREDICT_RANGE],
                    AnalysisParamsEnName.RANDOM_SEED: [AnalysisParamsDefault.RANDOM_SEED],
                    AnalysisParamsEnName.TIMESTEP: [AnalysisParamsDefault.TIMESTEP],
                    AnalysisParamsEnName.WEIGHT_DECAY: [AnalysisParamsDefault.WEIGHT_DECAY],
                },

                "time_scale_mark": False,
                "time_scale": [],

                "null_value_mark": True,
                "null_value": [AnalysisNullValueEnName.FILL_NULL_WITH_ZERO],

                "target_value_mark": True,
                "target_value": [AnalysisTargetValueEnName.ALL_BINARIZE],

                "normalize_value_mark": True,
                "normalize_value": [AnalysisNormalizeValueEnName.Z_SCORE],

                "sample_value_mark": False,
                "sample_value": [],

                "hyper_params_optimize_mark": True,
                "hyper_params_optimize": [AnalysisHyperParamsOptimizeEnName.BAYESIAN_OPTIMIZE],
                "hyper_params_optimize_trials": 5
            },
            "model_setting": {

                AnalysisModelEnName.ContrastiveBRAE: {
                    "negative_model": {
                        AnalysisModelEnName.AttentionVAE: {
                            "hidden_dim2": {"type": "int", "start": 20, "end": 30},
                            "hidden_dim3": {"type": "int", "start": 10, "end": 20},
                            "hidden_dim4": {"type": "int", "start": 3, "end": 10}
                        }
                    },
                    "positive_model": {
                        AnalysisModelEnName.BayesianRegression: {
                            "alpha": {"type": "float", "start": 0, "end": 1},
                            "beta": {"type": "float", "start": 0, "end": 1}
                        }
                    },
                    "threshold_setting": {
                        ContrastiveBRAEName.ADAPTIVE_THRESHOLD_ADJUSTMENT: {
                            "feature_threshold_ratio": {"type": "float", "start": 0, "end": 1}
                        }
                    }
                },

            }
        }

        return Resp.build_success(data={
            "defaultDataAnalysisConfig": default_data_analysis_config
        })
