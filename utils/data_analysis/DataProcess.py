import numpy as np
import pandas as pd
import torch
from collections import Counter
from tqdm import tqdm

from torch.utils.data import TensorDataset

from utils.common.GraphDataProcess import GraphDataProcess
from entity.dataloader.STDataLoader import STDataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pickle


class DataProcess:

    @classmethod
    def keep_specific_target(cls, df, target_name, target_value):
        new_df = df[df[target_name] == target_value].copy()
        return new_df

    @classmethod
    def del_na(cls, df):
        return df.dropna()

    @classmethod
    def mean_fill_na(cls, df):
        for col in df.columns:
            if df[col].isnull().any():  # 检查是否有缺失值
                mean_value = df[col].mean()  # 计算均值
                df[col] = df[col].fillna(mean_value)  # 填充缺失值

        return df

    @classmethod
    def enlarge_time_scale(cls, df, time_window):
        new_list = []
        for i in range(len(df) // time_window):
            if i < len(df) // time_window - 1:
                new_list.append(df.iloc[i * time_window: (i + 1) * time_window, :].mean())
            else:
                new_list.append(df.iloc[i * time_window:, :].mean())

        new_df = pd.DataFrame([dict(item) for item in new_list])
        new_df["target"] = new_df["target"].apply(lambda x: 1 if x > 0 else 0)

        return new_df

    @classmethod
    def box_statistics_enlarge_time_scale(cls, df: pd.DataFrame, time_scale: int):

        df.index = [x for x in range(len(df))]

        new_df_dict = {}
        for feature in tqdm(df.columns.values):
            if feature == "target":

                piece_list = []
                for i in range(int(len(df) / time_scale) + 1):
                    if i == int(len(df) / time_scale):
                        piece = df.loc[i * time_scale:, feature].tolist()
                    else:
                        piece = df.loc[i * time_scale: (i + 1) * time_scale, feature].tolist()

                    counter = {k: v for k, v in dict(Counter(piece)).items() if k != 0}

                    if counter != {}:
                        # TODO 可能出现多个相同数量的最大故障类型，怎么解决?
                        sorted_list = sorted([(k, v) for k, v in counter.items()], key=lambda v: v[1], reverse=True)
                        piece_list.append(sorted_list[0][0])
                    else:
                        piece_list.append(0)

                new_df_dict[f"{feature}"] = piece_list
            else:
                piece_dict = {}

                piece_dict["min"] = [df.loc[i * time_scale:, feature].min() if i == int(len(df) / time_scale) else df.loc[i * time_scale: (i + 1) * time_scale, feature].min() for i in range(int(len(df) / time_scale) + 1)]
                piece_dict["Q1"] = [df.loc[i * time_scale:, feature].quantile(0.25) if i == int(len(df) / time_scale) else df.loc[i * time_scale: (i + 1) * time_scale, feature].quantile(0.25) for i in range(int(len(df) / time_scale) + 1)]
                piece_dict["median"] = [df.loc[i * time_scale:, feature].median() if i == int(len(df) / time_scale) else df.loc[i * time_scale: (i + 1) * time_scale, feature].median() for i in range(int(len(df) / time_scale) + 1)]
                piece_dict["Q3"] = [df.loc[i * time_scale:, feature].quantile(0.75) if i == int(len(df) / time_scale) else df.loc[i * time_scale: (i + 1) * time_scale, feature].quantile(0.75) for i in range(int(len(df) / time_scale) + 1)]
                piece_dict["max"] = [df.loc[i * time_scale:, feature].max() if i == int(len(df) / time_scale) else df.loc[i * time_scale: (i + 1) * time_scale, feature].max() for i in range(int(len(df) / time_scale) + 1)]
                piece_dict["IQR"] = [piece_dict["Q3"][l] - piece_dict["Q1"][l] for l in range(len(piece_dict["Q3"]))]

                for k, v in piece_dict.items():
                    new_df_dict[f"{feature}_{k}"] = v

        new_df = pd.DataFrame(new_df_dict)

        return new_df

    @classmethod
    def split_tensor_into_positive_and_negative(cls, x_tensor: torch.Tensor, y_tensor: torch.Tensor):

        positive_x = []
        positive_y = []
        negative_x = []
        negative_y = []
        for i in range(len(y_tensor)):
            if y_tensor[i] == 1:
                positive_x.append(x_tensor[i, :, :])
                positive_y.append(y_tensor[i])
            else:
                negative_x.append(x_tensor[i, :, :])
                negative_y.append(y_tensor[i])

        positive_x_tensor = torch.stack(positive_x, 0)
        positive_y_tensor = torch.stack(positive_y, 0)
        negative_x_tensor = torch.stack(negative_x, 0)
        negative_y_tensor = torch.stack(negative_y, 0)

        return positive_x_tensor, positive_y_tensor, negative_x_tensor, negative_y_tensor

    @classmethod
    def fill_na(cls, df: pd.DataFrame, value=0) -> pd.DataFrame:

        df = df.fillna(value)

        return df

    @classmethod
    def drop_nan(cls, df: pd.DataFrame) -> pd.DataFrame:

        df = df.dropna()

        return df

    @classmethod
    def binarize_label(cls, df: pd.DataFrame, target_name: str, label_num) -> pd.DataFrame:

        df[target_name] = df[target_name].map(dict(zip([i for i in range(label_num + 1)], [0] + [1] * label_num)))

        return df

    @classmethod
    def multi_labels_z_score_normalize_feature(cls, df: pd.DataFrame, target_name: str, base_path, study_id, run_id) -> pd.DataFrame:

        x_col_list = df.columns.values.tolist()

        scaler = StandardScaler()
        array = scaler.fit_transform(df.loc[:, x_col_list])
        df = pd.DataFrame(array, columns=x_col_list)
        # 保存标准化器
        with open(f"{base_path}/{study_id}/{run_id}/scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        return df

    @classmethod
    def z_score_normalize_feature(cls, df: pd.DataFrame, target_name: str, base_path, study_id, run_id) -> pd.DataFrame:

        x_col_list = df.columns.values.tolist()
        x_col_list = [x for x in x_col_list if x not in [target_name, "timestamp", "wt_id"]]

        scaler = StandardScaler()
        array = scaler.fit_transform(df.loc[:, x_col_list])
        new_df = pd.DataFrame(array, columns=x_col_list)
        new_df[target_name] = df[target_name]
        # 保存标准化器
        with open(f"{base_path}/{study_id}/{run_id}/scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        return new_df

    @classmethod
    def multi_labels_min_max_normalize_feature(cls, df: pd.DataFrame, target_name: str, base_path, study_id, run_id) -> pd.DataFrame:

        x_col_list = df.columns.values.tolist()

        scaler = MinMaxScaler()
        array = scaler.fit_transform(df.loc[:, x_col_list])
        df = pd.DataFrame(array, columns=x_col_list)
        # 保存标准化器
        with open(f"{base_path}/{study_id}/{run_id}/scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        return df

    @classmethod
    def min_max_normalize_feature(cls, df: pd.DataFrame, target_name: str, base_path, study_id, run_id) -> pd.DataFrame:

        x_col_list = df.columns.values.tolist()
        x_col_list = [x for x in x_col_list if x not in [target_name, "timestamp", "wt_id"]]

        scaler = MinMaxScaler()
        array = scaler.fit_transform(df.loc[:, x_col_list])
        new_df = pd.DataFrame(array, columns=x_col_list)
        new_df[target_name] = df[target_name]
        # 保存标准化器
        with open(f"{base_path}/{study_id}/{run_id}/scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        return new_df

    @classmethod
    def multi_labels_split_df_data_into_x_with_y(cls, df: pd.DataFrame, target_name: str):

        # 多变量回归的X不包含目标变量
        data_x = df.loc[:, [col for col in df.columns.values.tolist() if col not in target_name.split(",")]]

        data_y = df.loc[:, target_name.split(",")]

        data_x = np.array(data_x)
        data_y = np.array(data_y)

        return data_x, data_y

    @classmethod
    def split_df_data_into_x_with_y(cls, df: pd.DataFrame, target_name: str):

        data_x = df
        data_y = df.loc[:, target_name]

        data_x = np.array(data_x)
        data_y = np.array(data_y)

        return data_x, data_y

    @classmethod
    def three_dim_split_df_data_into_x_with_y(cls, time_array: np.ndarray, target_name: str):

        data_x = time_array
        data_y = time_array[:, :, 0]

        return data_x, data_y

    @classmethod
    def split_time_series_data(cls, data_x: np.ndarray, data_y: np.ndarray, train_ratio: float, timestep: int, predict_range: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):

        train_size = int(train_ratio * len(data_x))

        train_data_x = data_x[: train_size, :]
        train_data_y = data_y[: train_size]
        test_data_x = data_x[train_size:, :]
        test_data_y = data_y[train_size:]

        train_time_data_x = []
        train_time_data_y = []
        test_time_data_x = []
        test_time_data_y = []

        for index in range(len(train_data_x) - timestep - predict_range + 1):
            train_time_data_x.append(train_data_x[index: index + timestep])
            train_time_data_y.append(train_data_y[index + timestep + predict_range - 1])

        for index in range(len(test_data_x) - timestep - predict_range + 1):
            test_time_data_x.append(test_data_x[index: index + timestep])
            test_time_data_y.append(test_data_y[index + timestep + predict_range - 1])

        x_train = np.array(train_time_data_x)
        y_train = np.array(train_time_data_y)
        x_test = np.array(test_time_data_x)
        y_test = np.array(test_time_data_y)

        return x_train, y_train, x_test, y_test

    @classmethod
    def three_dim_split_time_series_data(cls, data_x: np.ndarray, data_y: np.ndarray, train_ratio: float, timestep: int, predict_range: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):

        train_size = int(train_ratio * len(data_x))

        train_data_x = data_x[: train_size, :, :]
        train_data_y = data_y[: train_size, :]
        test_data_x = data_x[train_size:, :, :]
        test_data_y = data_y[train_size:, :]

        train_time_data_x = []
        train_time_data_y = []
        test_time_data_x = []
        test_time_data_y = []

        for index in range(len(train_data_x) - timestep):
            if index + timestep + predict_range < len(train_data_x):
                train_time_data_x.append(train_data_x[index: index + timestep, :, :])
                train_time_data_y.append(train_data_y[index + timestep + predict_range, :])

        for index in range(len(test_data_x) - timestep):
            if index + timestep + predict_range < len(test_data_x):
                test_time_data_x.append(test_data_x[index: index + timestep, :, :])
                test_time_data_y.append(test_data_y[index + timestep + predict_range, :])

        splited_train_time_data_x = []
        for array in train_time_data_x:
            splited_train_time_data_x.append(
                [x.reshape(array.shape[0], array.shape[2]) for x in np.split(array, array.shape[1], axis=1)]
            )
        x_train = []
        for i in range(data_x.shape[1]):
            x_train.append(np.array([x[i] for x in splited_train_time_data_x], dtype=np.float32))

        splited_train_time_data_y = []
        for array in train_time_data_y:
            splited_train_time_data_y.append(
                array.tolist()
            )
        y_train = []
        for i in range(data_y.shape[1]):
            y_train.append(np.array([y[i] for y in splited_train_time_data_y], dtype=np.float32))

        splited_test_time_data_x = []
        for array in test_time_data_x:
            splited_test_time_data_x.append(
                [x.reshape(array.shape[0], array.shape[2]) for x in np.split(array, array.shape[1], axis=1)]
            )
        x_test = []
        for i in range(data_x.shape[1]):
            x_test.append(np.array([x[i] for x in splited_test_time_data_x], dtype=np.float32))

        splited_test_time_data_y = []
        for array in test_time_data_y:
            splited_test_time_data_y.append(
                array.tolist()
            )
        y_test = []
        for i in range(data_y.shape[1]):
            y_test.append(np.array([y[i] for y in splited_test_time_data_y], dtype=np.float32))

        return x_train, y_train, x_test, y_test

    @classmethod
    def convert_data_from_ndarray_to_tensor(cls, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

        x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
        y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
        x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
        y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

        return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor

    @classmethod
    def three_dim_convert_data_from_ndarray_to_tensor(cls, x_train: list, y_train: list, x_test: list, y_test: list) -> (list, list, list, list):

        x_train_tensor = [torch.from_numpy(m).to(torch.float32) for m in x_train]
        y_train_tensor = [torch.from_numpy(m).to(torch.float32) for m in y_train]
        x_test_tensor = [torch.from_numpy(m).to(torch.float32) for m in x_test]
        y_test_tensor = [torch.from_numpy(m).to(torch.float32) for m in y_test]

        return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor

    @classmethod
    def get_dataloader_from_positive_negative_tensor(cls,
                                                     positive_x_train_tensor: torch.Tensor,
                                                     positive_y_train_tensor: torch.Tensor,
                                                     negative_x_train_tensor: torch.Tensor,
                                                     negative_y_train_tensor: torch.Tensor,
                                                     positive_x_test_tensor: torch.Tensor,
                                                     positive_y_test_tensor: torch.Tensor,
                                                     negative_x_test_tensor: torch.Tensor,
                                                     negative_y_test_tensor: torch.Tensor,
                                                     batch_size: int
                                                     ) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):

        positive_train_data = TensorDataset(positive_x_train_tensor, positive_y_train_tensor)
        negative_train_data = TensorDataset(negative_x_train_tensor, negative_y_train_tensor)

        positive_test_data = TensorDataset(positive_x_test_tensor, positive_y_test_tensor)
        negative_test_data = TensorDataset(negative_x_test_tensor, negative_y_test_tensor)

        positive_train_loader = torch.utils.data.DataLoader(positive_train_data, batch_size, True)
        negative_train_loader = torch.utils.data.DataLoader(negative_train_data, batch_size, True)

        positive_test_loader = torch.utils.data.DataLoader(positive_test_data, batch_size, False)
        negative_test_loader = torch.utils.data.DataLoader(negative_test_data, batch_size, False)

        return positive_train_loader, negative_train_loader, positive_test_loader, negative_test_loader

    @classmethod
    def get_dataloader_from_tensor(cls, x_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor, x_test_tensor: torch.Tensor, y_test_tensor: torch.Tensor, batch_size: int) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):

        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        test_data = TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size, True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size, False)

        return train_loader, test_loader

    @classmethod
    def three_dim_get_dataloader_from_tensor(cls, x_train_tensor: list, y_train_tensor: list, x_test_tensor: list, y_test_tensor: list, batch_size: int, random_seed: int) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):

        train_loader = STDataLoader(x_train_tensor, y_train_tensor, batch_size, True, random_seed)
        test_loader = STDataLoader(x_test_tensor, y_test_tensor, batch_size, False, random_seed)

        return train_loader, test_loader

    @classmethod
    def three_dim_multi_df_align_time(cls, id_df_dict: dict):

        total_df = pd.concat(id_df_dict).reset_index()
        total_df = total_df.groupby("timestamp")

        time_df_dict = {}
        for time, group_df in tqdm(total_df, desc="multi_df_align_time"):

            if len(group_df) == len(id_df_dict.keys()):

                time_df_dict[time] = group_df.sort_values("wt_id").reset_index().drop(
                    ["level_0", "level_1", "index", "timestamp", "wt_id"],
                    axis=1
                )

        time_array = np.stack([df.to_numpy() for df in time_df_dict.values()], axis=0)

        return time_array

    @classmethod
    def get_time_context_adj(cls, time_array: np.ndarray, time_context_adj_k: int):

        return GraphDataProcess.get_time_context_adj(
            time_array=time_array,
            time_context_adj_k=time_context_adj_k
        )

    @classmethod
    def get_distance_adj(cls, turbine_to_delete: list, distance_adj_k: int):

        return GraphDataProcess.get_distance_adj(
            turbine_to_delete=turbine_to_delete,
            distance_adj_k=distance_adj_k
        )

    @classmethod
    def convert_adj_into_index_form(cls, adj: np.ndarray):

        adj_index = []
        for i in range(len(adj)):

            for j in range(len(adj)):

                if adj[i, j] == 1:

                    adj_index.append([i, j])

        adj_index = np.array(adj_index).reshape(2, -1)

        return adj_index

    @classmethod
    def adj_convert_data_from_ndarray_to_tensor(cls, adj_array: np.ndarray) -> torch.Tensor:

        adj_tensor = torch.from_numpy(adj_array).to(torch.int8)

        return adj_tensor


