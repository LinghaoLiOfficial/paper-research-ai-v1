import numpy as np


class DeviationCalculator:

    @classmethod
    def get_label_from_deviation(cls, array: np.ndarray, threshold) -> np.ndarray:
        array = np.mean(array, axis=1)
        array = np.mean(array, axis=1)

        mapping_func = np.vectorize(lambda x: 1 if x >= threshold else 0)

        out = mapping_func(array)

        return out

    @classmethod
    def get_label_from_multi_deviation_train(cls, array, y, threshold, threshold_classification_model) -> np.ndarray:
        array = np.concatenate(array, axis=0)
        y = np.concatenate(y, axis=0)

        array = np.mean(array, axis=1)

        y_array_pred_list = []
        for i, feature in enumerate(threshold):
            mapping_func = np.vectorize(lambda x: 1 if x >= feature else 0)

            feature_array = array[:, i]

            out = mapping_func(feature_array)
            y_array_pred_list.append(out)

        featured_array = np.stack(y_array_pred_list, axis=1)

        pred = threshold_classification_model.train(featured_array, y)

        # 计算每个值的各个特征有几个是偏离的

        # out_list = []
        # for j in range(featured_array.shape[0]):
        #     current_value = featured_array[j, :].reshape(-1)
        #
        #     deviation_percentage = np.sum(current_value == 1) / len(current_value)
        #     out_list.append(1 if deviation_percentage >= feature_threshold_ratio else 0)
        #
        # pred = np.array(out_list)

        return threshold_classification_model

    @classmethod
    def get_label_from_multi_deviation(cls, array: np.ndarray, threshold, threshold_classification_model, feature_threshold_ratio) -> np.ndarray:
        array = np.mean(array, axis=1)
        # array = array[:, -1, :]

        y_array_pred_list = []
        for i, feature in enumerate(threshold):
            mapping_func = np.vectorize(lambda x: 1 if x >= feature else 0)

            feature_array = array[:, i]

            out = mapping_func(feature_array)
            y_array_pred_list.append(out)

        featured_array = np.stack(y_array_pred_list, axis=1)

        pred = threshold_classification_model.test(featured_array)

        # 计算每个值的各个特征有几个是偏离的

        # out_list = []
        # for j in range(featured_array.shape[0]):
        #     current_value = featured_array[j, :].reshape(-1)
        #
        #     deviation_percentage = np.sum(current_value == 1) / len(current_value)
        #     out_list.append(1 if deviation_percentage >= feature_threshold_ratio else 0)
        #
        # pred = np.array(out_list)

        return pred

