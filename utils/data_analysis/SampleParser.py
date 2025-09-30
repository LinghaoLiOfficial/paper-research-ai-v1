import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
import pywt


class SampleParser:

    @classmethod
    def down_sampling_label_target(cls, seq, time_range):

        new_seq = []
        for i in range(int(len(seq) / time_range) + 1):

            if i == int(len(seq) / time_range):
                seq_range = seq[i * time_range:]
            else:
                seq_range = seq[i * time_range: (i + 1) * time_range]

            avg_seq = sum(seq_range) / len(seq_range)

            avg_seq = 1 if avg_seq > 0 else 0

            new_seq.append(avg_seq)

        new_seq = np.array(new_seq)

        return new_seq

    # 小波变换
    @classmethod
    def wavelet_transform(cls, df, frequency=0, wavelet="db4", level=4):

        # 填充信号长度到2的幂次

        padding_len = 2 ** int(np.ceil(np.log2(len(df)))) - len(df)

        padding_array = np.array([0] * padding_len)

        y = np.append(padding_array, df.loc[:, "target"].values)

        coeffs = []
        for i in range(df.shape[1]):

            wave = pywt.wavedec(df.iloc[:, i].values, wavelet, level=level)

            coeffs.append(wave)

        approximations = [coeff[frequency] for coeff in coeffs]

        time_range = int(len(df) / len(approximations[0]))

        y = cls.down_sampling_label_target(
            seq=y,
            time_range=time_range
        )

        new_df = np.array([y] + approximations)

        new_df = pd.DataFrame(
            data=new_df,
            columns=df.columns.values.tolist()
        )

        return new_df

    @classmethod
    def shuffle_x_with_y(cls, x, y, random_seed):

        np.random.seed(random_seed)

        indices = np.random.permutation(x.shape[0])

        new_x = x[indices]
        new_y = y[indices]

        return new_x, new_y

    @classmethod
    def smote_over_sampling(cls, x_train: np.ndarray, y_train: np.ndarray, random_seed) -> (np.ndarray, np.ndarray):

        x_size, timestep, feature_size = x_train.shape

        x_train = x_train.reshape(x_size, -1)

        smote = SMOTE(
            sampling_strategy=1,
            random_state=random_seed
        )

        new_x_train, new_y_train = smote.fit_resample(
            X=x_train,
            y=y_train
        )

        new_x_train = new_x_train.reshape(-1, timestep, feature_size)

        new_x_train, new_y_train = cls.shuffle_x_with_y(
            x=new_x_train,
            y=new_y_train,
            random_seed=random_seed
        )

        return new_x_train, new_y_train

    @classmethod
    def check_if_positive_sample(cls, tensor: torch.Tensor) -> bool:

        # 二分类
        if 0 in tensor:
            return False
        return True

    @classmethod
    def check_if_negative_sample(cls, tensor: torch.Tensor) -> bool:

        # 二分类
        if 1 in tensor:
            return False
        return True

    @classmethod
    def three_dim_check_if_negative_sample(cls, y_train: list) -> bool:

        # 二分类
        for m in y_train:
            if 1 in m:
                return False
        return True

    @classmethod
    def filter_positive_sample(cls, x_train: np.ndarray, y_train: np.ndarray) -> (np.ndarray, np.ndarray):

        new_x_train = []
        new_y_train = []
        for i in range(len(y_train)):

            if y_train[i].item() == 1:
                new_x_train.append(x_train[i, :, :])
                new_y_train.append(y_train[i])

        new_x_train = np.array(new_x_train)
        new_y_train = np.array(new_y_train)

        return new_x_train, new_y_train
