import matplotlib

from config.name.data_analysis.AnalysisTaskTypeEnName import AnalysisTaskTypeEnName
from entity.machine_learning_model.KernelBayesianRegression import KernelBayesianRegression

# 指定使用AGG后端，避免GUI线程冲突
matplotlib.use('Agg')  # 在导入pyplot之前设置
import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import networkx as nx
import pandas as pd

from entity.machine_learning_model.BayesianRegression import BayesianRegression
from utils.common.MyColor import MyColor
import copy
import os.path

import optuna
from tqdm import tqdm
import torch
import numpy as np

from config.model.LossFunctionNameStr import LossFunctionNameStr
from config.model.OptimizerNameStr import OptimizerNameStr
from config.name.data_analysis.AnalysisModelEnName import AnalysisModelEnName
from config.name.data_analysis.module.ContrastiveBRAEName import ContrastiveBRAEName
from entity.LossFunctionFactory import LossFunctionFactory
from entity.ModelFactory import ModelFactory
from entity.OptimizerFactory import OptimizerFactory
from mapper.DataAnalysisMapper import DataAnalysisMapper
from utils.common.DeviationCalculator import DeviationCalculator
from utils.data_analysis.BestModelParser import BestModelParser
from utils.data_analysis.indicator.ClassificationIndicatorsParser import ClassificationIndicatorsParser
from utils.data_analysis.DataProcess import DataProcess
from utils.data_analysis.DataVisualize import DataVisualize
from utils.data_analysis.HyperParamsParser import HyperParamsParser
from utils.common.RandomStrGenerator import RandomStrGenerator
from utils.data_analysis.TemporalModelTrain import TemporalModelTrain
from utils.data_analysis.TrainingHistorySaver import TrainingHistorySaver
from utils.data_analysis.indicator.RegressionIndicatorsParser import RegressionIndicatorsParser


class ContrastiveBRAEPattern:
    @classmethod
    def run(cls, task, trail: optuna.trial):

        # 分割训练集tensor数据变为正样本训练集tensor数据+负样本训练集tensor数据
        positive_x_train_tensor, positive_y_train_tensor, negative_x_train_tensor, negative_y_train_tensor = DataProcess.split_tensor_into_positive_and_negative(
            x_tensor=task.x_train_tensor,
            y_tensor=task.y_train_tensor
        )

        # 分割测试集tensor数据变为正样本测试集tensor数据+负样本测试集tensor数据
        positive_x_test_tensor, positive_y_test_tensor, negative_x_test_tensor, negative_y_test_tensor = DataProcess.split_tensor_into_positive_and_negative(
            x_tensor=task.x_test_tensor,
            y_tensor=task.y_test_tensor
        )

        # 根据正样本训练集观测tensor数据生成正样本训练集观测ndarray数据
        positive_x_train_array = positive_x_train_tensor.detach().cpu().numpy()

        positive_train_loader, negative_train_loader, positive_test_loader, negative_test_loader = DataProcess.get_dataloader_from_positive_negative_tensor(
            positive_x_train_tensor=positive_x_train_tensor,
            positive_y_train_tensor=positive_y_train_tensor,
            negative_x_train_tensor=negative_x_train_tensor,
            negative_y_train_tensor=negative_y_train_tensor,
            positive_x_test_tensor=positive_x_test_tensor,
            positive_y_test_tensor=positive_y_test_tensor,
            negative_x_test_tensor=negative_x_test_tensor,
            negative_y_test_tensor=negative_y_test_tensor,
            batch_size=task.task_params['batch_size']
        )

        best_model_parser = BestModelParser(
            user_id=task.user_id,
            task_id=task.task_id
        )

        # 实例化模型

        negative_model_config = HyperParamsParser.parse(HyperParamsParser.get_value(task.model_config[ContrastiveBRAEName.NEGATIVE_MODEL]), trail)
        negative_model_name = HyperParamsParser.get_key(task.model_config[ContrastiveBRAEName.NEGATIVE_MODEL])
        negative_model = ModelFactory.get_model(
            model_name=negative_model_name,
            device=task.device,
            hyper_params=negative_model_config
        )
        task.chosen_data_analysis_config['model_setting'][task.model_name][ContrastiveBRAEName.NEGATIVE_MODEL] = {negative_model_name: negative_model_config}

        positive_model_config = HyperParamsParser.parse(HyperParamsParser.get_value(task.model_config[ContrastiveBRAEName.POSITIVE_MODEL]), trail)
        positive_model_name = HyperParamsParser.get_key(task.model_config[ContrastiveBRAEName.POSITIVE_MODEL])
        positive_model = ModelFactory.get_model(
            model_name=positive_model_name,
            device=task.device,
            hyper_params=positive_model_config
        )
        task.chosen_data_analysis_config['model_setting'][task.model_name][ContrastiveBRAEName.POSITIVE_MODEL] = {positive_model_name: positive_model_config}

        threshold_setting_model_config = HyperParamsParser.parse(HyperParamsParser.get_value(task.model_config[ContrastiveBRAEName.THRESHOLD_SETTING]), trail)
        threshold_setting_model = ModelFactory.get_model(
            model_name=AnalysisModelEnName.DecisionTree,
            device=task.device,
            hyper_params=threshold_setting_model_config
        )
        task.chosen_data_analysis_config['model_setting'][task.model_name][ContrastiveBRAEName.THRESHOLD_SETTING] = {AnalysisModelEnName.DecisionTree: threshold_setting_model_config}

        # 实例化损失函数和优化器

        negative_loss_function = LossFunctionFactory.get_loss_function(LossFunctionNameStr.CrossEntropyLoss)
        negative_optimizer = OptimizerFactory.get_optimizer(OptimizerNameStr.Adam, negative_model, task.task_params)

        run_folder_path = f"{task.base_path}/{task.study_counter}/{task.run_counter}"
        if not os.path.exists(run_folder_path):
            os.mkdir(run_folder_path)

        # 增加模型训练硬件信息
        current_data_analysis_config = copy.deepcopy(task.chosen_data_analysis_config)
        current_data_analysis_config['device'] = str(task.device)

        # 迭代训练测试模型

        train_loss = []
        test_loss = []
        for epoch in range(task.task_params['epochs']):
            # 训练负样本自监督模型+计算负样本自监督模型的训练损失值
            negative_train_indicators_result, negative_model, negative_loss_function, negative_optimizer, loss = cls.deep_learning_self_supervised_model_train(
                model=negative_model,
                model_name=negative_model_name,
                loss_function=negative_loss_function,
                optimizer=negative_optimizer,
                train_loader=negative_train_loader,
                hyper_params=task.task_params,
                epoch=epoch,
                device=task.device
            )
            train_loss.append(loss)
            print("negative_train_indicators_result: {}\n".format(negative_train_indicators_result))

            # 计算负样本自监督模型的测试损失值
            loss = cls.new_deep_learning_self_supervised_model_test(
                model=negative_model,
                model_name=negative_model_name,
                loss_function=negative_loss_function,
                test_loader=negative_train_loader,
                hyper_params=task.task_params,
                epoch=epoch,
                device=task.device
            )
            test_loss.append(loss)

            # 如果为第一次迭代，则训练正样本自监督模型+计算正样本自监督模型的训练损失值
            if epoch == 0:
                positive_train_indicators_result, positive_model = cls.regression_machine_learning_supervised_model_train(
                    model=positive_model,
                    x=positive_x_train_array
                )
                print("positive_train_indicators_result: {}\n".format(positive_train_indicators_result))

            # 根据正样本测试数据集，计算正样本偏差、正样本正偏差、正样本负偏差
            positive_all_deviation, positive_all_positive_deviation, positive_all_negative_deviation = cls.calculate_deviation(
                positive_model=positive_model,
                negative_model=negative_model,
                negative_model_name=negative_model_name,
                loader=positive_test_loader,
                hyper_params=task.task_params,
                epoch=epoch,
                device=task.device
            )

            # 根据负样本测试数据集，计算负样本偏差、负样本正偏差、负样本负偏差
            negative_all_deviation, negative_all_positive_deviation, negative_all_negative_deviation = cls.calculate_deviation(
                positive_model=positive_model,
                negative_model=negative_model,
                negative_model_name=negative_model_name,
                loader=negative_test_loader,
                hyper_params=task.task_params,
                epoch=epoch,
                device=task.device
            )

            # 根据正样本偏差、负样本偏差，计算异常值的阈值
            anomaly_threshold_list = cls.calculate_anomaly_threshold(
                positive_all_deviation=positive_all_deviation,
                negative_all_deviation=negative_all_deviation
            )

            # 训练阈值分类模型
            threshold_classification_model = cls.deep_learning_contrastive_self_supervised_model_eval_train(
                positive_model=positive_model,
                negative_model=negative_model,
                negative_model_name=negative_model_name,
                train_loader=task.train_loader,
                hyper_params=task.task_params,
                epoch=epoch,
                device=task.device,
                anomaly_threshold_list=anomaly_threshold_list,
                threshold_classification_model=threshold_setting_model
            )

            # 测试阈值分类模型
            test_indicators_result = cls.deep_learning_contrastive_self_supervised_model_eval(
                positive_model=positive_model,
                negative_model=negative_model,
                negative_model_name=negative_model_name,
                test_loader=task.test_loader,
                hyper_params=task.task_params,
                epoch=epoch,
                device=task.device,
                anomaly_threshold_list=anomaly_threshold_list,
                threshold_classification_model=threshold_classification_model,
                feature_threshold_ratio=task.task_params['train_ratio']
            )
            print("test_indicators_result: {}\n".format(test_indicators_result))

            # 早停策略【准确率不能过高】
            if test_indicators_result["accuracy"] > 0.99:
                break

            history_id = RandomStrGenerator.generate_uuid()

            # 保存单次迭代最佳数据到实例
            best_model_parser.compare_bigger_save(
                indicator="f1",
                indicators_result=test_indicators_result,
                model=negative_model,
                epoch=epoch,
                params={
                    "anomaly_threshold_list": anomaly_threshold_list,
                    "positive_all_deviation": positive_all_deviation,
                    "negative_all_deviation": negative_all_deviation
                },
                history_id=history_id,
                study_id=task.study_counter,
                run_id=task.run_counter
            )

            best_indicators_result = best_model_parser.display_result()
            print("best_indicators_result: {}\n".format(best_indicators_result))

            # 保存单次迭代数据到数据库
            mysql_result = DataAnalysisMapper.insert_classification_training_history({
                "history_id": history_id,
                "history_owner": task.user_id,
                "task_id": task.task_id,
                "task_name": task.task_name,
                "study_id": task.study_counter,
                "run_id": task.run_counter,
                "history_epoch": epoch + 1,
                "train_loss": negative_train_indicators_result['loss'],
                "train_accuracy": negative_train_indicators_result['accuracy'],
                "train_recall": negative_train_indicators_result['recall'],
                "train_precision": negative_train_indicators_result['precision'],
                "train_f1": negative_train_indicators_result['f1'],
                "train_other": "",
                "test_loss": test_indicators_result['loss'],
                "test_accuracy": test_indicators_result['accuracy'],
                "test_recall": test_indicators_result['recall'],
                "test_precision": test_indicators_result['precision'],
                "test_f1": test_indicators_result['f1'],
                "test_other": "",
                "history_params": str(current_data_analysis_config)
            })

        # 查看单次优化最佳指标结果
        best_indicators_result = best_model_parser.display_result()
        print("run_best_indicators_result: {}\n".format(best_indicators_result))

        mysql_result1 = DataAnalysisMapper.insert_best_classification_training_history({
            "history_id": best_indicators_result['history_id'],
            "history_owner": task.user_id,
            "task_id": task.task_id,
            "task_name": task.task_name,
            "study_id": task.study_counter,
            "run_id": task.run_counter,
            "history_epoch": best_indicators_result['epoch'],
            "test_loss": best_indicators_result['loss'],
            "test_accuracy": best_indicators_result['accuracy'],
            "test_recall": best_indicators_result['recall'],
            "test_precision": best_indicators_result['precision'],
            "test_f1": best_indicators_result['f1'],
            "test_other": "",
            "history_params": str(current_data_analysis_config)
        })

        DataVisualize.new_draw_anomaly_plots(
            anomaly_threshold_list=best_model_parser.best_params['anomaly_threshold_list'],
            positive_all_deviation=best_model_parser.best_params['positive_all_deviation'],
            negative_all_deviation=best_model_parser.best_params['negative_all_deviation'],
            study_id=task.study_counter,
            run_id=task.run_counter,
            base_path=task.base_path
        )

        # cls.nll_plot_train_test_curve_with_variance(
        #     model=positive_model,
        #     X=positive_x_train_array,
        #     y=positive_x_train_array,
        #     study_id=task.study_counter,
        #     run_id=task.run_counter,
        #     base_path=task.base_path,
        #     task=task
        # )

        DataVisualize.loss_plot_train_test_curve_with_variance(
            train_loss=train_loss,
            test_loss=test_loss,
            study_id=task.study_counter,
            run_id=task.run_counter,
            base_path=task.base_path
        )

        return best_indicators_result["accuracy"], best_indicators_result["recall"], best_indicators_result["precision"]\

    @classmethod
    def nll_plot_train_test_curve_with_variance(cls, model, X, y, study_id, run_id, base_path, task, num_samples=10):
        plt.clf()

        if len(X.shape) == 3:
            X = X.reshape(-1, X.shape[2])

        if isinstance(model, BayesianRegression):
            train_model = BayesianRegression(
                alpha=model.alpha,
                beta=model.beta
            )
        elif isinstance(model, KernelBayesianRegression):
            train_model = KernelBayesianRegression(
                alpha=model.alpha,
                beta=model.beta,
                length_scale=model.length_scale
            )

        data_sizes = np.arange(50, len(X), 60)
        nll_means = []
        nll_stds = []

        for size in tqdm(data_sizes, desc="draw nll curve"):

            nll_samples = []

            # 使用多次采样来计算均值和方差
            for _ in range(num_samples):
                # 随机选择样本子集来进行拟合和计算
                indices = np.random.choice(np.arange(size), size=size, replace=True)
                X_subset = X[:size][indices]
                train_model.train(X_subset)
                nll = train_model.log_likelihood(X_subset, X_subset)
                nll_samples.append(nll)

            # 计算每个数据集大小下的NLL均值和标准差
            nll_means.append(np.mean(nll_samples))
            nll_stds.append(np.std(nll_samples))

        nll_means = np.array(nll_means)
        nll_stds = np.array(nll_stds)

        # 绘制NLL均值曲线和方差区间

        plt.figure(figsize=(8, 6), dpi=300)

        scale_factor = 10

        plt.plot(data_sizes, nll_means, label="train", color=MyColor.COLOR_1)
        plt.fill_between(data_sizes, nll_means - scale_factor * nll_stds, nll_means + scale_factor * nll_stds,
                         color=MyColor.COLOR_1, alpha=0.2)

        nll_means = []
        nll_stds = []

        for size in tqdm(data_sizes, desc="draw nll curve"):
            nll_samples = []

            # 使用多次采样来计算均值和方差
            for _ in range(num_samples):
                # 随机选择样本子集来进行拟合和计算
                indices = np.random.choice(np.arange(size), size=size, replace=True)
                X_subset = X[:size][indices]
                nll = model.log_likelihood(X_subset, X_subset)
                nll_samples.append(nll)

            # 计算每个数据集大小下的NLL均值和标准差
            nll_means.append(np.mean(nll_samples))
            nll_stds.append(np.std(nll_samples))

        nll_means = np.array(nll_means)
        nll_stds = np.array(nll_stds)

        plt.plot(data_sizes, nll_means, label="val", color=MyColor.COLOR_3)
        plt.fill_between(data_sizes, nll_means - scale_factor * nll_stds, nll_means + scale_factor * nll_stds,
                         color=MyColor.COLOR_3, alpha=0.2)

        # plt.title("Negative Log Likelihood (NLL) with Variance vs Data Size")
        plt.xlabel("Data Size", fontsize=14, labelpad=16)
        plt.ylabel("Negative Log Likelihood", fontsize=14, labelpad=16)
        plt.legend(loc="upper right", fontsize=14)
        # plt.show()

        plt.savefig(f"{base_path}/{study_id}/{run_id}/positive_model.svg")

    @classmethod
    def deep_learning_self_supervised_model_train(cls, model, model_name, loss_function, optimizer, train_loader, hyper_params, epoch, device):

        model.train()

        train_indicators_parser = ClassificationIndicatorsParser()

        first_loss = float("inf")
        last_loss = float("inf")
        loss_list = []
        train_bar = tqdm(train_loader)
        for index, train_tensors in enumerate(train_bar):

            x_train, y_train = train_tensors

            x_train = x_train.to(device)
            y_train = y_train.to(device)

            optimizer.zero_grad()

            if model_name in ["VAE", "RVAE", "AttentionVAE"]:

                x_train_pred, kl_divergence = model(x_train)

                loss = loss_function(x_train_pred, x_train) + kl_divergence

            else:

                x_train_pred = model(x_train)

                loss = loss_function(x_train_pred, x_train)

            loss.backward()

            optimizer.step()

            train_loss = train_indicators_parser.calculate_loss(
                loss=loss.item()
            )

            train_bar.desc = "train epoch[{}/{}] (first:{:4f}, last:{:4f}) loss:{:.4f}".format(
                epoch + 1,
                hyper_params["epochs"],
                first_loss,
                last_loss,
                train_loss
            )

            if index == 1:
                first_loss = train_loss
            elif index == len(train_loader) - 2:
                last_loss = train_loss

            loss_list.append(train_loss)

        train_indicators_result = train_indicators_parser.display_result()

        return train_indicators_result, model, loss_function, optimizer, loss_list

    @classmethod
    def new_deep_learning_self_supervised_model_test(cls, model, model_name, loss_function, test_loader,
                                                     hyper_params, epoch, device):

        model.eval()

        test_indicators_parser = ClassificationIndicatorsParser()

        with torch.no_grad():
            first_loss = float("inf")
            last_loss = float("inf")
            loss_list = []
            test_bar = tqdm(test_loader)
            for index, test_tensors in enumerate(test_bar):

                x_test, y_test = test_tensors

                x_test = x_test.to(device)
                y_test = y_test.to(device)

                if model_name in ["VAE", "RVAE", "AttentionVAE"]:

                    x_test_pred, kl_divergence = model(x_test)

                    loss = loss_function(x_test_pred, x_test) + kl_divergence

                else:

                    x_test_pred = model(x_test)

                    loss = loss_function(x_test_pred, x_test)

                test_loss = test_indicators_parser.calculate_loss(
                    loss=loss.item()
                )

                test_bar.desc = "test epoch[{}/{}] (first:{:4f}, last:{:4f}) loss:{:.4f}".format(
                    epoch + 1,
                    hyper_params["epochs"],
                    first_loss,
                    last_loss,
                    test_loss
                )

                if index == 1:
                    first_loss = test_loss
                elif index == len(test_loader) - 2:
                    last_loss = test_loss

                loss_list.append(test_loss)

        test_indicators_result = test_indicators_parser.display_result()

        return loss_list

    @classmethod
    def regression_machine_learning_supervised_model_train(cls, model, x):

        train_indicators_parser = RegressionIndicatorsParser()

        x_train_pred = model.train(x)

        mse = train_indicators_parser.calculate_mse(x, x_train_pred)

        train_indicators_result = train_indicators_parser.display_result()

        return train_indicators_result, model

    @classmethod
    def calculate_deviation(cls, positive_model, negative_model, negative_model_name, loader, hyper_params, epoch, device):

        with torch.no_grad():

            all_positive_deviation_list = []
            all_negative_deviation_list = []
            all_deviation_list = []
            bar = tqdm(loader)
            for index, tensors in enumerate(bar):

                x_train, y_train = tensors

                x_train = x_train.to(device)
                y_train = y_train.to(device)

                print(x_train.shape)

                if negative_model_name in ["VAE", "RVAE", "AttentionVAE"]:
                    negative_x_pred, _ = negative_model(x_train)
                else:
                    negative_x_pred = negative_model(x_train)
                positive_x_pred = positive_model.test(x_train.detach().cpu().numpy())

                x_train = x_train.detach().cpu().numpy()
                negative_x_pred = negative_x_pred.detach().cpu().numpy()
                # positive_x_pred = positive_x_pred.detach().cpu().numpy()

                negative_deviation = np.abs(negative_x_pred - x_train)
                positive_deviation = np.abs(positive_x_pred - x_train)

                deviation = (negative_deviation - positive_deviation)

                # TODO

                # 法1: 计算历史序列中的各权重 [a^0 + a^1 + a^2 + a^3 + a^4 + ... = 1]

                # 考虑序列数据的时间价值递减，指数衰减法

                # decay = 0.8
                #
                # weight_num = deviation.shape[1]
                # weight_array = decay ** np.arange(weight_num)
                # weight_array = weight_array
                #
                # deviation = deviation * weight_array[:, np.newaxis]
                # negative_deviation = negative_deviation * weight_array[:, np.newaxis]
                # positive_deviation = positive_deviation * weight_array[:, np.newaxis]

                deviation = np.mean(deviation, axis=1)
                negative_deviation = np.mean(negative_deviation, axis=1)
                positive_deviation = np.mean(positive_deviation, axis=1)

                # deviation = deviation[:, -1, :]
                # negative_deviation = negative_deviation[:, -1, :]
                # positive_deviation = positive_deviation[:, -1, :]

                all_deviation_list.append(deviation)
                all_negative_deviation_list.append(negative_deviation)
                all_positive_deviation_list.append(positive_deviation)

            return all_deviation_list, all_positive_deviation_list, all_negative_deviation_list

    @classmethod
    def calculate_anomaly_threshold(cls, positive_all_deviation, negative_all_deviation):

        # 法1: 百分位数法
        # anomaly_threshold = np.percentile(all_deviation, hyper_params["anomaly_percentile"])

        # 法2: 均值加标准差法

        # mean_error = np.mean(all_deviation)
        # std_error = np.std(all_deviation)

        # 中位数法

        anomaly_threshold_list = []
        feature_size = negative_all_deviation[0].shape[1]
        for feature in range(feature_size):

            negative_deviation = []
            for i in range(len(negative_all_deviation)):
                negative_deviation.append(negative_all_deviation[i][:, feature].reshape(-1))

            positive_deviation = []
            for i in range(len(positive_all_deviation)):
                positive_deviation.append(positive_all_deviation[i][:, feature].reshape(-1))

            positive_deviation = np.concatenate(positive_deviation, axis=0)
            negative_deviation = np.concatenate(negative_deviation, axis=0)

            positive_median = np.median(positive_deviation)
            negative_median = np.median(negative_deviation)

            anomaly_threshold = (positive_median + negative_median) / 2
            anomaly_threshold_list.append(anomaly_threshold)

        return anomaly_threshold_list

    @classmethod
    def deep_learning_contrastive_self_supervised_model_eval_train(cls, positive_model, negative_model, negative_model_name, train_loader, hyper_params, epoch, device, anomaly_threshold_list, threshold_classification_model):

        negative_model.eval()

        train_indicators_parser = ClassificationIndicatorsParser()

        deviation_list = []
        y_train_list = []

        with torch.no_grad():

            train_bar = tqdm(train_loader)
            for index, train_tensors in enumerate(train_bar):

                x_train, y_train = train_tensors

                x_train = x_train.to(device)
                y_train = y_train.to(device)

                if negative_model_name in ["VAE", "RVAE", "AttentionVAE"]:
                    negative_x_train_pred, _ = negative_model(x_train)
                else:
                    negative_x_train_pred = negative_model(x_train)
                positive_x_train_pred = positive_model.train(x_train.detach().cpu().numpy())

                positive_x_train_pred = torch.from_numpy(positive_x_train_pred).to(torch.float32).to(device)

                negative_deviation = torch.abs(negative_x_train_pred - x_train)
                positive_deviation = torch.abs(positive_x_train_pred - x_train)

                deviation = (negative_deviation - positive_deviation)

                deviation = deviation.detach().cpu().numpy()

                y_train = y_train.detach().cpu().numpy()

                deviation_list.append(deviation)
                y_train_list.append(y_train)

                train_bar.desc = "train epoch[{}/{}]".format(
                    epoch + 1,
                    hyper_params["epochs"]
                )

            threshold_classification_model = DeviationCalculator.get_label_from_multi_deviation_train(
                array=deviation_list,
                y=y_train_list,
                threshold=anomaly_threshold_list,
                threshold_classification_model=threshold_classification_model
            )

            train_indicators_result = train_indicators_parser.display_result()

            return threshold_classification_model

    @classmethod
    def deep_learning_contrastive_self_supervised_model_eval(cls, positive_model, negative_model, negative_model_name, test_loader, hyper_params, epoch, device, anomaly_threshold_list, threshold_classification_model, feature_threshold_ratio):

        negative_model.eval()

        test_indicators_parser = ClassificationIndicatorsParser()

        with torch.no_grad():

            test_bar = tqdm(test_loader)
            for index, test_tensors in enumerate(test_bar):

                x_test, y_test = test_tensors

                x_test = x_test.to(device)
                y_test = y_test.to(device)

                if negative_model_name in ["VAE", "RVAE", "AttentionVAE"]:
                    negative_x_test_pred, _ = negative_model(x_test)
                else:
                    negative_x_test_pred = negative_model(x_test)
                positive_x_test_pred = positive_model.test(x_test.detach().cpu().numpy())

                positive_x_test_pred = torch.from_numpy(positive_x_test_pred).to(torch.float32).to(device)

                negative_deviation = torch.abs(negative_x_test_pred - x_test)
                positive_deviation = torch.abs(positive_x_test_pred - x_test)

                deviation = (negative_deviation - positive_deviation)

                deviation = deviation.detach().cpu().numpy()

                y_test = y_test.detach().cpu().numpy()

                y_test_pred = DeviationCalculator.get_label_from_multi_deviation(
                    array=deviation,
                    threshold=anomaly_threshold_list,
                    threshold_classification_model=threshold_classification_model,
                    feature_threshold_ratio=feature_threshold_ratio
                )

                accuracy = test_indicators_parser.calculate_accuracy(
                    real_array=y_test,
                    pred_array=y_test_pred
                )

                precision = test_indicators_parser.calculate_precision(
                    real_array=y_test,
                    pred_array=y_test_pred
                )

                recall = test_indicators_parser.calculate_recall(
                    real_array=y_test,
                    pred_array=y_test_pred
                )

                f1 = test_indicators_parser.calculate_f1(
                    real_array=y_test,
                    pred_array=y_test_pred
                )

                balanced_accuracy_with_recall = test_indicators_parser.calculate_balanced_accuracy_with_recall(
                    accuracy=accuracy,
                    recall=recall
                )

                test_bar.desc = "test epoch[{}/{}] acc:{:.4f} precision:{:4f} recall:{:4f} f1:{:4f} balanced_accuracy_with_recall:{:4f}".format(
                    epoch + 1,
                    hyper_params["epochs"],
                    accuracy,
                    precision,
                    recall,
                    f1,
                    balanced_accuracy_with_recall
                )

            test_indicators_result = test_indicators_parser.display_result()

            return test_indicators_result
