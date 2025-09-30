import copy
import os.path

import optuna
from tqdm import tqdm
import torch
import numpy as np

from config.model.LossFunctionNameStr import LossFunctionNameStr
from config.model.OptimizerNameStr import OptimizerNameStr
from config.name.data_analysis.module.ContrastiveBRAEName import ContrastiveBRAEName
from config.name.data_analysis.module.SingleModelName import SingleModelName
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


class WaveNetPattern:
    @classmethod
    def run(cls, task, trail: optuna.trial):

        train_loader, test_loader = DataProcess.get_dataloader_from_tensor(
            x_train_tensor=task.x_train_tensor,
            y_train_tensor=task.y_train_tensor,
            x_test_tensor=task.x_test_tensor,
            y_test_tensor=task.y_test_tensor,
            batch_size=task.task_params['batch_size']
        )

        best_model_parser = BestModelParser(
            user_id=task.user_id,
            task_id=task.task_id
        )

        # 实例化模型

        model_config = HyperParamsParser.parse(HyperParamsParser.get_value(task.model_config[SingleModelName.MODEL]), trail)
        model_name = HyperParamsParser.get_key(task.model_config[SingleModelName.MODEL])
        model = ModelFactory.get_model(
            model_name=model_name,
            device=task.device,
            hyper_params=model_config
        )
        task.chosen_data_analysis_config['model_setting'][task.model_name][SingleModelName.MODEL] = {model_name: model_config}

        # 实例化损失函数和优化器

        loss_function = LossFunctionFactory.get_loss_function(LossFunctionNameStr.MSELoss)
        optimizer = OptimizerFactory.get_optimizer(OptimizerNameStr.Adam, model, task.task_params)

        run_folder_path = f"{task.base_path}/{task.study_counter}/{task.run_counter}"
        if not os.path.exists(run_folder_path):
            os.mkdir(run_folder_path)

        # 增加模型训练硬件信息
        current_data_analysis_config = task.data_analysis_config
        current_data_analysis_config['device'] = str(task.device)

        # 迭代训练测试模型

        train_loss = []
        test_loss = []
        for epoch in range(task.task_params['epochs']):
            # 模型训练
            train_indicators_result, model, loss_function, optimizer, loss = cls.model_train(
                model=model,
                model_name=model_name,
                loss_function=loss_function,
                optimizer=optimizer,
                train_loader=train_loader,
                hyper_params=task.task_params,
                epoch=epoch,
                device=task.device
            )
            train_loss.append(loss)
            print("train_indicators_result: {}\n".format(train_indicators_result))

            # 模型测试
            test_indicators_result, loss = cls.model_test(
                model=model,
                model_name=model_name,
                loss_function=loss_function,
                optimizer=optimizer,
                test_loader=test_loader,
                hyper_params=task.task_params,
                epoch=epoch,
                device=task.device
            )
            test_loss.append(loss)
            print("test_indicators_result: {}\n".format(test_indicators_result))

            history_id = RandomStrGenerator.generate_uuid()

            # 保存单次迭代最佳数据到实例
            best_model_parser.compare_smaller_save(
                indicator="rmse",
                indicators_result=test_indicators_result,
                model=model,
                epoch=epoch,
                params={},
                history_id=history_id,
                study_id=task.study_counter,
                run_id=task.run_counter
            )

            best_indicators_result = best_model_parser.display_result()
            print("best_indicators_result: {}\n".format(best_indicators_result))

            # 保存单次迭代数据到数据库
            mysql_result = DataAnalysisMapper.insert_regression_training_history({
                "history_id": history_id,
                "history_owner": task.user_id,
                "task_id": task.task_id,
                "task_name": task.task_name,
                "study_id": task.study_counter,
                "run_id": task.run_counter,
                "history_epoch": epoch + 1,
                "train_loss": train_indicators_result['loss'],
                "train_mse": train_indicators_result['mse'],
                "train_rmse": train_indicators_result['rmse'],
                "train_mae": train_indicators_result['mae'],
                "train_r2": train_indicators_result['r2'],
                "train_other": "",
                "test_loss": test_indicators_result['loss'],
                "test_mse": test_indicators_result['mse'],
                "test_rmse": test_indicators_result['rmse'],
                "test_mae": test_indicators_result['mae'],
                "test_r2": test_indicators_result['r2'],
                "test_other": "",
                "history_params": str(current_data_analysis_config)
            })

            # 查看单次优化最佳指标结果
        best_indicators_result = best_model_parser.display_result()
        print("run_best_indicators_result: {}\n".format(best_indicators_result))

        mysql_result1 = DataAnalysisMapper.insert_best_regression_training_history({
            "history_id": best_indicators_result['history_id'],
            "history_owner": task.user_id,
            "task_id": task.task_id,
            "task_name": task.task_name,
            "study_id": task.study_counter,
            "run_id": task.run_counter,
            "history_epoch": best_indicators_result['epoch'],
            "test_loss": best_indicators_result['loss'],
            "test_mse": best_indicators_result['mse'],
            "test_rmse": best_indicators_result['rmse'],
            "test_mae": best_indicators_result['mae'],
            "test_r2": best_indicators_result['r2'],
            "test_other": "",
            "history_params": str(current_data_analysis_config)
        })

        DataVisualize.loss_plot_train_test_curve_with_variance(
            train_loss=train_loss,
            test_loss=test_loss,
            study_id=task.study_counter,
            run_id=task.run_counter,
            base_path=task.base_path
        )

        return best_indicators_result["rmse"], best_indicators_result["mae"], best_indicators_result["r2"]

    @classmethod
    def model_train(cls, model, model_name, loss_function, optimizer, train_loader, hyper_params, epoch, device):

        model.train()

        train_indicators_parser = RegressionIndicatorsParser()

        first_loss = float("inf")
        last_loss = float("inf")
        loss_list = []
        train_bar = tqdm(train_loader)
        for index, train_tensors in enumerate(train_bar):
            x_train, y_train = train_tensors

            x_train = x_train.to(device)
            y_train = y_train.to(device)

            optimizer.zero_grad()
            x_train_pred = model(x_train)
            loss = loss_function(x_train_pred, y_train)

            loss.backward()
            optimizer.step()

            y_train = y_train.detach().cpu().numpy()
            x_train_pred = x_train_pred.detach().cpu().numpy()

            train_loss = train_indicators_parser.calculate_loss(
                loss=loss.item()
            )

            mse = train_indicators_parser.calculate_mse(
                real_array=y_train,
                pred_array=x_train_pred
            )

            rmse = train_indicators_parser.calculate_rmse(
                real_array=y_train,
                pred_array=x_train_pred
            )

            mae = train_indicators_parser.calculate_mae(
                real_array=y_train,
                pred_array=x_train_pred
            )

            r2 = train_indicators_parser.calculate_r2(
                real_array=y_train,
                pred_array=x_train_pred
            )

            adjusted_r2 = train_indicators_parser.calculate_adjusted_r2(
                real_array=y_train,
                pred_array=x_train_pred,
                p=x_train.shape[2]
            )

            mape = train_indicators_parser.calculate_mape(
                real_array=y_train,
                pred_array=x_train_pred
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
    def model_test(cls, model, model_name, loss_function, optimizer, test_loader, hyper_params, epoch, device):

        model.eval()

        test_indicators_parser = RegressionIndicatorsParser()

        first_loss = float("inf")
        last_loss = float("inf")
        loss_list = []
        test_bar = tqdm(test_loader)
        for index, test_tensors in enumerate(test_bar):
            x_test, y_test = test_tensors

            x_test = x_test.to(device)
            y_test = y_test.to(device)

            optimizer.zero_grad()
            x_test_pred = model(x_test)
            loss = loss_function(x_test_pred, y_test)

            y_test = y_test.detach().cpu().numpy()
            x_test_pred = x_test_pred.detach().cpu().numpy()

            test_loss = test_indicators_parser.calculate_loss(
                loss=loss.item()
            )

            mse = test_indicators_parser.calculate_mse(
                real_array=y_test,
                pred_array=x_test_pred
            )

            rmse = test_indicators_parser.calculate_rmse(
                real_array=y_test,
                pred_array=x_test_pred
            )

            mae = test_indicators_parser.calculate_mae(
                real_array=y_test,
                pred_array=x_test_pred
            )

            r2 = test_indicators_parser.calculate_r2(
                real_array=y_test,
                pred_array=x_test_pred
            )

            adjusted_r2 = test_indicators_parser.calculate_adjusted_r2(
                real_array=y_test,
                pred_array=x_test_pred,
                p=x_test.shape[2]
            )

            mape = test_indicators_parser.calculate_mape(
                real_array=y_test,
                pred_array=x_test_pred
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

        return test_indicators_result, loss_list
