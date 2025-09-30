import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from factor_analyzer import calculate_kmo


if __name__ == '__main__':
    """
        线性可分性验证
    """

    df_list = []
    base_path = './data/classification/time_series/su_you'
    for file_name in os.listdir(base_path):
        df = pd.read_csv(f"{base_path}/{file_name}")
        df_list.append(df)
        del df

        # if len(df_list) >= 6:
        #     break

    total_df = pd.concat(df_list, axis=0)
    del df_list

    target_name = 'target'

    # 过滤正常数据
    total_df = total_df[total_df[target_name] != 0]

    data_x = total_df
    data_y = total_df.loc[:, target_name]

    data_x = np.array(data_x)
    data_y = np.array(data_y)

    kmo_all, kmo_model = calculate_kmo(data_x)
    print(f"KMO统计量: {kmo_model:.3f}")  # >0.6适合PCA

    # 标准化
    linear_validation_scaler = StandardScaler()
    data_x_scaled = linear_validation_scaler.fit_transform(data_x)

    # PCA线性降维
    linear_validation_pca = PCA(n_components=2)
    data_x_pca = linear_validation_pca.fit_transform(data_x_scaled)
    print(f"主成分方差贡献率: {linear_validation_pca.explained_variance_ratio_}")

    # 绘制PCA降维图
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data_x_pca[:, 0], data_x_pca[:, 1], c=data_y, cmap='viridis', alpha=0.7, s=1)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)

    plt.show()