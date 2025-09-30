import copy

import optuna
import matplotlib

from config.name.data_analysis.AnalysisHyperParamsOptimizeEnName import AnalysisHyperParamsOptimizeEnName
from config.name.data_analysis.AnalysisTaskTypeEnName import AnalysisTaskTypeEnName
from utils.data_analysis.DataVisualize import DataVisualize

# 指定使用AGG后端，避免GUI线程冲突
matplotlib.use('Agg')  # 在导入pyplot之前设置
import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt
import pandas as pd

from config.name.data_analysis.AnalysisNormalizeValueEnName import AnalysisNormalizeValueEnName
from config.name.data_analysis.AnalysisNullValueEnName import AnalysisNullValueEnName
from config.name.data_analysis.AnalysisParamsEnName import AnalysisParamsEnName
from config.name.data_analysis.AnalysisTargetValueEnName import AnalysisTargetValueEnName
from pattern.ModelPatternFactory import ModelPatternFactory
from utils.data_analysis.DataImport import DataImport
from utils.data_analysis.DataProcess import DataProcess
from utils.data_analysis.DataViewer import DataViewer
from utils.data_analysis.DeviceParser import DeviceParser
import itertools


"""
    模型训练结构:
    - task
        - study
            - run
                - epoch
"""


class ModelTrainingTask:
    def __init__(self, data_analysis_config):
        self.data_analysis_config = data_analysis_config
        self.chosen_data_analysis_config = copy.deepcopy(data_analysis_config)
        self.device = DeviceParser.load_device()
        self.user_id = self.data_analysis_config['user_id']
        self.task_id = self.data_analysis_config['task_id']
        self.task_name = self.data_analysis_config['task_name']
        self.base_path = self.data_analysis_config['base_path']

        self.dataset_name = None
        self.dataset_config = None
        self.deal_null_value_method = None
        self.deal_normalize_value_method = None
        self.model_name = None
        self.model_config = None
        self.task_params = None
        self.task_type = None
        self.data_type = None
        self.target_name = None

        self.df = None
        self.data_x = None
        self.data_y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_train_tensor = None
        self.y_train_tensor = None
        self.x_test_tensor = None
        self.y_test_tensor = None
        self.train_loader = None
        self.test_loader = None

        self.study_counter = -1
        self.run_counter = -1

    def start(self):
        # 创建多个study
        for dataset_name, dataset_config in self.data_analysis_config['data_setting']['dataset'].items():
            self.dataset_name = dataset_name
            self.dataset_config = dataset_config
            self.chosen_data_analysis_config['data_setting']['dataset'] = {self.dataset_name: self.dataset_config}
            if self.data_analysis_config['data_setting']['null_value_mark']:
                for deal_null_value_method in self.data_analysis_config['data_setting']['null_value']:
                    self.iterate_deal_normalize_value_method(dataset_name, dataset_config, deal_null_value_method)
            else:
                self.iterate_deal_normalize_value_method(dataset_name, dataset_config)

    def iterate_deal_normalize_value_method(self, dataset_name, dataset_config, deal_null_value_method=None):
        self.deal_null_value_method = deal_null_value_method
        self.chosen_data_analysis_config['data_setting']['null_value'] = self.deal_null_value_method
        if self.data_analysis_config['data_setting']['normalize_value_mark']:
            for deal_normalize_value_method in self.data_analysis_config['data_setting']['normalize_value']:
                self.iterate_model_setting(dataset_name, dataset_config, deal_null_value_method, deal_normalize_value_method)
        else:
            self.iterate_model_setting(dataset_name, dataset_config, deal_null_value_method)

    def iterate_model_setting(self, dataset_name, dataset_config, deal_null_value_method, deal_normalize_value_method=None):
        self.deal_normalize_value_method = deal_normalize_value_method
        self.chosen_data_analysis_config['data_setting']['normalize_value'] = self.deal_normalize_value_method
        for model_name, model_config in self.data_analysis_config['model_setting'].items():
            self.model_name = model_name

            module_name_list = [x for x in model_config.keys()]
            module_config_list = [x for x in model_config.values()]

            # 将每个字典转换为键值对的列表
            dicts_as_kv_lists = [list(d.items()) for d in module_config_list]
            # 使用 itertools.product 生成所有可能的组合
            combinations = itertools.product(*dicts_as_kv_lists)
            # 将每个组合转换为字典
            combinations_dict = [dict(comb) for comb in combinations]

            # 定义每个键对应的字段映射
            key_to_field = {module_name: [x for x in model_config[module_name].keys()] for module_name in module_name_list}

            # 转换函数
            def transform_data(keys, data, key_to_field):
                result = []
                for item in data:
                    new_dict = {}
                    for key in keys:
                        # 获取当前键对应的字段
                        fields = key_to_field[key]
                        # 提取字段对应的值
                        values = {field: item[field] for field in fields if field in item}
                        new_dict[key] = values
                    result.append(new_dict)
                return result

            # 调用函数并输出结果
            new_model_config_list = transform_data(module_name_list, combinations_dict, key_to_field)

            for new_model_config in new_model_config_list:
                self.model_config = new_model_config
                self.chosen_data_analysis_config['model_setting'] = {self.model_name: self.model_config}

                # 获取所有键和对应的值列表
                keys = list(self.data_analysis_config['data_setting']['task_params'].keys())
                values = list(self.data_analysis_config['data_setting']['task_params'].values())

                # 使用 itertools.product 生成所有可能的组合
                params_combinations = itertools.product(*values)

                # 将每个组合转换为字典
                task_params_list = [dict(zip(keys, comb)) for comb in params_combinations]

                for task_params in task_params_list:
                    self.task_params = task_params
                    self.chosen_data_analysis_config['data_setting']['task_params'] = self.task_params

                    self.task_type = self.data_analysis_config['data_setting']['task_type']
                    self.data_type = self.data_analysis_config['data_setting']['data_type']
                    self.target_name = self.dataset_config['target_name']

                    self.start_study()

    def start_study(self):

        self.run_counter = -1
        self.study_counter += 1

        # 若不存在则创建文件夹
        if not os.path.exists(f"{self.base_path}/{self.study_counter}"):
            os.makedirs(f"{self.base_path}/{self.study_counter}")

        if self.data_analysis_config['data_setting']['hyper_params_optimize_mark']:
            if self.data_analysis_config['data_setting']['hyper_params_optimize'][0] == AnalysisHyperParamsOptimizeEnName.BAYESIAN_OPTIMIZE:
                # 创建多个run
                if self.data_analysis_config['data_setting']['task_type'] == AnalysisTaskTypeEnName.CLASSIFICATION:
                    study = optuna.create_study(study_name="study", directions=["maximize", "maximize", "maximize"])
                    obj_names = ["accuracy", "recall", "precision"]
                elif self.data_analysis_config['data_setting']['task_type'] == AnalysisTaskTypeEnName.REGRESSION:
                    study = optuna.create_study(study_name="study", directions=["minimize", "minimize", "maximize"])
                    obj_names = ["rmse", "mae", "r2"]

                study.optimize(self.start_run, n_trials=self.data_analysis_config['data_setting']['hyper_params_optimize_trials'])
                print("best_trials: {}".format(study.best_trials))

                trial_list = []
                study_trials = study.get_trials()
                for trial in study_trials:
                    trial_dict = {}
                    trial_dict.update(trial.params)
                    trial_dict[obj_names[0]] = trial.values[0]
                    trial_dict[obj_names[1]] = trial.values[1]
                    trial_dict[obj_names[2]] = trial.values[2]

                    trial_list.append(trial_dict)

                study_df_col = [x for x in trial_list[0].keys()]
                study_df_data = [[y for y in x.values()] for x in trial_list]
                study_df = pd.DataFrame(data=study_df_data, columns=study_df_col)

                if not os.path.exists(f"{self.base_path}/general"):
                    os.makedirs(f"{self.base_path}/general")

                study_df.to_csv(f"{self.base_path}/general/study.csv", index=False)

                # 各个超参数对结果影响的重要性
                plt.clf()
                optuna.visualization.matplotlib.plot_param_importances(study)
                plt.savefig(f"{self.base_path}/general/param_importance.svg", dpi=300)

                # 绘制帕累托前沿二维投影图
                plt.clf()
                DataVisualize.draw_pareto_projection_plot(
                    table_path=f"{self.base_path}/general/study.csv",
                    task_type=self.task_type,
                    save_path=f"{self.base_path}/general/pareto.svg"
                )

                # # 在trail中每次的objective value和当前的最优解
                # plt.clf()
                # optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[0], target_name="accuracy").show()
                # optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[1], target_name="recall").show()
                # # 每个超参数在所有trail中取值的分布，以散点图的形式
                # plt.clf()
                # optuna.visualization.plot_slice(study, target=lambda t: t.values[0], target_name="accuracy").show()
                # optuna.visualization.plot_slice(study, target=lambda t: t.values[1], target_name="recall").show()
                #
                # # 位于帕累托前沿上的trials
                # plotly.io.show(optuna.visualization.plot_pareto_front(study, target_names=["accuracy", "recall"]))
                # optuna.visualization.matplotlib.plot_pareto_front(study, target_names=["accuracy", "recall"])
                #
                # plt.savefig('./data/{}.png'.format("多参数搜索"), dpi=300)

    def start_run(self, trail: optuna.trial):
        # 创建多个epoch

        self.run_counter += 1

        # 若不存在则创建文件夹
        if not os.path.exists(f"{self.base_path}/{self.study_counter}/{self.run_counter}"):
            os.makedirs(f"{self.base_path}/{self.study_counter}/{self.run_counter}")

        # 导入数据
        df = DataImport.import_data(data_path=f"./data/{self.task_type}/{self.data_type}/{self.dataset_name}", add_timestamp=False)

        # 处理缺失值
        if self.deal_null_value_method == AnalysisNullValueEnName.FILL_NULL_WITH_ZERO:
            df = DataProcess.fill_na(df=df, value=0)
        elif self.deal_null_value_method == AnalysisNullValueEnName.FILL_NULL_WITH_MEAN:
            df = DataProcess.mean_fill_na(df=df)
        elif self.deal_null_value_method == AnalysisNullValueEnName.DEL_NULL:
            df = DataProcess.del_na(df=df)

        # 处理目标值
        is_multi_labels = False
        if self.dataset_config['target_method']['target_method_name'] == AnalysisTargetValueEnName.ALL_BINARIZE:
            df = DataProcess.binarize_label(df=df, target_name=self.target_name, label_num=max(df[self.target_name]))
        elif self.dataset_config['target_method']['target_method_name'] == AnalysisTargetValueEnName.SPECIFIC_TARGET:
            df = DataProcess.keep_specific_target(df=df, target_name=self.target_name, target_value=self.dataset_config['target_method']['target_method_value'])
        elif self.dataset_config['target_method']['target_method_name'] == AnalysisTargetValueEnName.MULTI_LABELS:
            # 无需处理
            pass
        elif self.dataset_config['target_method']['target_method_name'] == AnalysisTargetValueEnName.SINGLE_VAR:
            # 无需处理
            pass
        elif self.dataset_config['target_method']['target_method_name'] == AnalysisTargetValueEnName.MULTI_VAR:
            is_multi_labels = True

        # 查看全量数据的目标 值标签数量占比
        if not is_multi_labels:
            print(f"view_label_num: {DataViewer.view_label_num(iteration=df[self.target_name])}\n")

        # 数据标准化
        if self.deal_normalize_value_method == AnalysisNormalizeValueEnName.MIN_MAX:
            if is_multi_labels:
                # 多变量回归也标准化目标值
                df = DataProcess.multi_labels_min_max_normalize_feature(df=df, target_name=self.target_name, base_path=self.base_path, study_id=self.study_counter, run_id=self.run_counter)
            else:
                df = DataProcess.min_max_normalize_feature(df=df, target_name=self.target_name, base_path=self.base_path, study_id=self.study_counter, run_id=self.run_counter)
        elif self.deal_normalize_value_method == AnalysisNormalizeValueEnName.Z_SCORE:
            if is_multi_labels:
                # 多变量回归也标准化目标值
                df = DataProcess.multi_labels_z_score_normalize_feature(df=df, target_name=self.target_name, base_path=self.base_path, study_id=self.study_counter, run_id=self.run_counter)
            else:
                df = DataProcess.z_score_normalize_feature(df=df, target_name=self.target_name, base_path=self.base_path, study_id=self.study_counter, run_id=self.run_counter)

        # 分割全量数据变为观测数据+目标数据【分割df变为data_x, data_y】
        if is_multi_labels:
            data_x, data_y = DataProcess.multi_labels_split_df_data_into_x_with_y(df=df, target_name=self.target_name)
        else:
            data_x, data_y = DataProcess.split_df_data_into_x_with_y(df=df, target_name=self.target_name)

        # 按比例分割观测数据+目标数据变为时间序列的训练集和测试集
        x_train, y_train, x_test, y_test = DataProcess.split_time_series_data(
            data_x=data_x,
            data_y=data_y,
            train_ratio=self.task_params['train_ratio'],
            timestep=self.task_params['timestep'],
            predict_range=self.task_params['predict_range']
        )

        if not is_multi_labels:
            # 分别查看训练集和测试集的目标值标签数量占比
            print(f"y_train: {DataViewer.view_label_num(iteration=y_train)}\n")
            print(f"y_test: {DataViewer.view_label_num(iteration=y_test)}\n")

        # 获取数据表的输入维度和输出维度
        for module_name, module_params in self.model_config.items():
            for model_name, model_params in module_params.items():
                self.model_config[module_name][model_name]["input_dim"] = data_x.shape[1]
                if is_multi_labels:
                    self.model_config[module_name][model_name]["output_dim"] = data_y.shape[1]

        # 将ndarray格式数据转换为tensor格式数据
        x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = DataProcess.convert_data_from_ndarray_to_tensor(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )

        # 生成dataLoader
        train_loader, test_loader = DataProcess.get_dataloader_from_tensor(
            x_train_tensor=x_train_tensor,
            y_train_tensor=y_train_tensor,
            x_test_tensor=x_test_tensor,
            y_test_tensor=y_test_tensor,
            batch_size=self.task_params['batch_size']
        )

        self.df = df
        self.data_x = data_x
        self.data_y = data_y
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_train_tensor = x_train_tensor
        self.y_train_tensor = y_train_tensor
        self.x_test_tensor = x_test_tensor
        self.y_test_tensor = y_test_tensor
        self.train_loader = train_loader
        self.test_loader = test_loader

        # 注册模型自定义模板
        model_pattern = ModelPatternFactory.get_model_pattern(self.data_analysis_config['data_setting']['task_pattern'])
        if model_pattern:
            return model_pattern.run(copy.deepcopy(self), trail)


