from tqdm import tqdm
import torch
import numpy as np

from config.name.data_analysis.AnalysisModelEnName import AnalysisModelEnName
from utils.data_analysis.indicator.ClassificationIndicatorsParser import ClassificationIndicatorsParser
from utils.common.DeviationCalculator import DeviationCalculator


class TemporalModelTrain:

    @classmethod
    def regression_machine_learning_supervised_model_train(cls, model, x):

        x_train_pred = model.train(x)

        mse = np.mean((x - x_train_pred) ** 2)

        print("mse: {:4f}".format(mse))

        return model, mse

    @classmethod
    def machine_learning_supervised_model_train(cls, model, model_name, x_train, y_train, x_test, y_test, hyper_params, epoch, device):

        train_indicators_parser = ClassificationIndicatorsParser()

        if model_name == AnalysisModelEnName.LightGBM:
            y_train_pred = model.train(x_train, y_train, x_test, y_test)

        else:
            y_train_pred = model.train(x_train, y_train)

        accuracy = train_indicators_parser.calculate_accuracy(
            real_array=y_train,
            pred_array=y_train_pred
        )

        precision = train_indicators_parser.calculate_precision(
            real_array=y_train,
            pred_array=y_train_pred
        )

        recall = train_indicators_parser.calculate_recall(
            real_array=y_train,
            pred_array=y_train_pred
        )

        f1 = train_indicators_parser.calculate_f1(
            real_array=y_train,
            pred_array=y_train_pred
        )

        balanced_accuracy_with_recall = train_indicators_parser.calculate_balanced_accuracy_with_recall(
            accuracy=accuracy,
            recall=recall
        )

        train_indicators_result = train_indicators_parser.display_result()

        return train_indicators_result, model

    @classmethod
    def machine_learning_supervised_model_eval(cls, model, model_name, x_test, y_test, hyper_params,
                                               epoch, device):

        test_indicators_parser = ClassificationIndicatorsParser()

        y_test_pred = model.test(x_test)

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

        test_indicators_result = test_indicators_parser.display_result()

        return test_indicators_result

    @classmethod
    def deep_learning_supervised_model_train(cls, model, model_name, loss_function, optimizer, train_loader, hyper_params, epoch, device):

        model.train()

        train_indicators_parser = ClassificationIndicatorsParser()

        first_loss = float("inf")
        last_loss = float("inf")
        train_bar = tqdm(train_loader)
        for index, train_tensors in enumerate(train_bar):

            x_train, y_train = train_tensors

            x_train = x_train.to(device)
            y_train = y_train.to(device)

            optimizer.zero_grad()

            y_train_pred = model(x_train)

            y_train = y_train.reshape(-1).long()

            loss = loss_function(y_train_pred, y_train)

            loss.backward()

            optimizer.step()

            y_train = y_train.detach().cpu().numpy()
            y_train_pred = y_train_pred.detach().cpu().numpy().argmax(axis=1)

            train_loss = train_indicators_parser.calculate_loss(
                loss=loss.item()
            )

            accuracy = train_indicators_parser.calculate_accuracy(
                real_array=y_train,
                pred_array=y_train_pred
            )

            precision = train_indicators_parser.calculate_precision(
                real_array=y_train,
                pred_array=y_train_pred
            )

            recall = train_indicators_parser.calculate_recall(
                real_array=y_train,
                pred_array=y_train_pred
            )

            f1 = train_indicators_parser.calculate_f1(
                real_array=y_train,
                pred_array=y_train_pred
            )

            balanced_accuracy_with_recall = train_indicators_parser.calculate_balanced_accuracy_with_recall(
                accuracy=accuracy,
                recall=recall
            )

            train_bar.desc = "train epoch[{}/{}] (first:{:4f}, last:{:4f}) acc:{:.4f} precision:{:4f} recall:{:4f} f1:{:4f} balanced_accuracy_with_recall:{:4f}".format(
                epoch + 1,
                hyper_params["epochs"],
                first_loss,
                last_loss,
                accuracy,
                precision,
                recall,
                f1,
                balanced_accuracy_with_recall
            )

            if index == 1:
                first_loss = train_loss
            elif index == len(train_loader) - 2:
                last_loss = train_loss

        train_indicators_result = train_indicators_parser.display_result()

        return train_indicators_result, model, loss_function, optimizer

    @classmethod
    def deep_learning_supervised_model_eval(cls, model, model_name, loss_function, test_loader,
                                            hyper_params, epoch, device):

        model.eval()

        test_indicators_parser = ClassificationIndicatorsParser()

        with torch.no_grad():

            first_loss = float("inf")
            last_loss = float("inf")
            test_bar = tqdm(test_loader)
            for index, test_tensors in enumerate(test_bar):

                x_test, y_test = test_tensors

                x_test = x_test.to(device)
                y_test = y_test.to(device)

                y_test_pred = model(x_test)

                y_test = y_test.reshape(-1).long()

                loss = loss_function(y_test_pred, y_test)

                y_test = y_test.detach().cpu().numpy()
                y_test_pred = y_test_pred.detach().cpu().numpy().argmax(axis=1)

                test_loss = test_indicators_parser.calculate_loss(
                    loss=loss.item()
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

                test_bar.desc = "test epoch[{}/{}] (first:{:4f}, last:{:4f}) acc:{:.4f} precision:{:4f} recall:{:4f} f1:{:4f} balanced_accuracy_with_recall:{:4f}".format(
                    epoch + 1,
                    hyper_params["epochs"],
                    first_loss,
                    last_loss,
                    accuracy,
                    precision,
                    recall,
                    f1,
                    balanced_accuracy_with_recall
                )

                if index == 1:
                    first_loss = test_loss
                elif index == len(test_loader) - 2:
                    last_loss = test_loss

            test_indicators_result = test_indicators_parser.display_result()

            return test_indicators_result

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
    def deep_learning_self_supervised_model_eval(cls, model, model_name, test_loader, hyper_params, epoch, device):

        model.eval()

        test_indicators_parser = ClassificationIndicatorsParser()

        with torch.no_grad():

            test_bar = tqdm(test_loader)
            for index, test_tensors in enumerate(test_bar):

                x_test, y_test = test_tensors

                x_test = x_test.to(device)
                y_test = y_test.to(device)

                if model_name in ["VAE", "RVAE", "AttentionVAE"]:
                    x_test_pred, _ = model(x_test)
                else:
                    x_test_pred = model(x_test)

                deviation = torch.abs(x_test_pred - x_test)

                deviation = deviation.detach().cpu().numpy()

                y_test = y_test.detach().cpu().numpy()

                y_test_pred = DeviationCalculator.get_label_from_deviation(
                    array=deviation,
                    threshold=hyper_params["threshold_value"]
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
