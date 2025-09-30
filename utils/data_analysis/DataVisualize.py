import matplotlib

from config.name.data_analysis.AnalysisTaskTypeEnName import AnalysisTaskTypeEnName

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


class DataVisualize:

    @classmethod
    def draw_community_graph(cls, nodes, edges, communities, name_mapping):
        plt.clf()

        # 创建有向图
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        cluster_colors = [communities[node] for node in G.nodes()]

        # 使用力引导布局
        pos = nx.spring_layout(
            G,
            k=0.2,
            seed=42
        )

        # 设置可视化参数
        node_size = 100
        edge_color = "gray"

        # 绘制节点
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=cluster_colors,
            cmap=plt.cm.tab10,  # 使用预定义颜色映射
            node_size=node_size,
            edgecolors="black"  # 节点边框颜色
        )

        # 绘制有向边
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_color,
            width=0.6,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=7,
            node_size=node_size
        )

        # 添加节点标签
        nx.draw_networkx_labels(
            G,
            pos,
            font_size=4,
            font_color="black",
            labels=name_mapping
        )

        plt.axis("off")

        plt.tight_layout()
        plt.savefig("./test.png", dpi=300)

    @classmethod
    def draw_pareto_projection_plot(cls, table_path, task_type, save_path):
        def identify_pareto(points):
            """
            识别帕累托前沿点，最大化目标1，最小化目标2
            :param points: (N, 2) 形状的numpy数组，每行是一个解的两个目标值
            :return: 帕累托前沿的布尔索引
            """
            is_pareto = np.ones(points.shape[0], dtype=bool)  # 假设所有点都是帕累托点

            for i, point in enumerate(points):
                if is_pareto[i]:
                    # 如果任何点的目标1比当前点大，且目标2比当前点小，则当前点被支配
                    is_pareto[i] = not np.any(
                        (points[:, 0] > point[0]) & (points[:, 1] > point[1]) & (points[:, 2] > point[2]))

            return is_pareto

        df = pd.read_csv(table_path)
        if task_type == AnalysisTaskTypeEnName.CLASSIFICATION:
            points = df.loc[:, ["accuracy", "recall", "precision"]].to_numpy()
        else:
            points = df.loc[:, ["rmse", "mae", "r2"]].to_numpy()

        # points = np.array([[0.73, 0.75, 0.32], [0, 0, 0]])

        # points[:, 0] = points[:, 0] * 1.1
        # points[:, 2] = points[:, 2] * 1.2

        # 识别帕累托前沿
        pareto_mask = identify_pareto(points)
        pareto_points = points[pareto_mask]

        fig = plt.figure(figsize=(8, 6), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        # # 绘制帕累托前沿图
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, color=MyColor.COLOR_3, alpha=1, label='solutions')  # 所有点
        # ax.scatter(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2], s=8, color=MyColor.COLOR_1, alpha=1, label='pareto front')  # 帕累托前沿

        # 绘制 xy 平面的投影并填充
        ax.scatter(0.02 + points[:, 0], 0.02 + points[:, 1], np.zeros_like(points[:, 2]),
                   color=MyColor.COLOR_1, alpha=0.3, s=10, label="solution xy")
        ax.scatter(0.02 + pareto_points[:, 0], 0.02 + pareto_points[:, 1], np.zeros_like(pareto_points[:, 2]),
                   color=MyColor.COLOR_1, alpha=0.3, s=10, edgecolors=MyColor.COLOR_5, label="pareto front xy")

        # 绘制 xz 平面的投影并填充
        ax.scatter(points[:, 0], np.zeros_like(points[:, 1]), points[:, 2],
                   color=MyColor.COLOR_4, alpha=0.3, s=10, label="solution xz")
        ax.scatter(pareto_points[:, 0], np.zeros_like(pareto_points[:, 1]), pareto_points[:, 2],
                   color=MyColor.COLOR_4, alpha=0.3, s=10, edgecolors=MyColor.COLOR_5, label="pareto front xz")

        # 绘制 yz 平面的投影并填充
        ax.scatter(np.zeros_like(points[:, 0]), points[:, 1], points[:, 2],
                   color=MyColor.COLOR_3, alpha=0.3, s=10, label="solution yz")
        ax.scatter(np.zeros_like(pareto_points[:, 0]), pareto_points[:, 1], pareto_points[:, 2],
                   color=MyColor.COLOR_3, alpha=0.3, s=10, edgecolors=MyColor.COLOR_5, label="pareto front yz")

        ax.legend(loc="best")

        # ax.set_title('Pareto Front (Maximize Objective 1, Minimize Objective 2)')
        if task_type == AnalysisTaskTypeEnName.CLASSIFICATION:
            ax.set_xlabel("accuracy", labelpad=12)
            ax.set_ylabel("recall", labelpad=12)
            ax.set_zlabel("precision", labelpad=12)
        else:
            ax.set_xlabel("rmse", labelpad=12)
            ax.set_ylabel("mae", labelpad=12)
            ax.set_zlabel("r2", labelpad=12)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        # ax.set_zlim([0, 1])

        ax.set_xticks(np.arange(0, 1.0, 0.1))
        ax.set_yticks(np.arange(0, 1.0, 0.1))
        # ax.set_zticks(np.arange(0, 1.0, 0.1))

        ax.view_init(azim=45)

        # 添加网格
        # ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        # plt.show()

        plt.savefig(save_path)

    @classmethod
    def loss_plot_train_test_curve_with_variance(cls, train_loss, test_loss, study_id, run_id, base_path):

        plt.clf()

        train_loss_mean = np.array([sum(x) / len(x) for x in train_loss])
        test_loss_mean = np.array([sum(x) / len(x) for x in test_loss])

        train_loss_std = np.array([np.std(np.array(x)) for x in train_loss])
        test_loss_std = np.array([np.std(np.array(x)) for x in test_loss])

        plt.figure(figsize=(8, 6), dpi=300)

        scale_factor = 1

        plt.plot(np.array([i for i in range(len(train_loss))]), train_loss_mean, label="train", color=MyColor.COLOR_1)
        plt.fill_between(np.array([i for i in range(len(train_loss))]), train_loss_mean - scale_factor * train_loss_std,
                         train_loss_mean + scale_factor * train_loss_std, color=MyColor.COLOR_1, alpha=0.2)

        plt.plot([i for i in range(len(test_loss))], test_loss_mean, label="val", color=MyColor.COLOR_3)
        plt.fill_between(np.array([i for i in range(len(test_loss))]), test_loss_mean - scale_factor * test_loss_std,
                         test_loss_mean + scale_factor * test_loss_std, color=MyColor.COLOR_3, alpha=0.2)

        # plt.title("Negative Log Likelihood (NLL) with Variance vs Data Size")
        plt.xlabel("Epoch", fontsize=14, labelpad=16)
        plt.ylabel("Loss", fontsize=14, labelpad=16)
        plt.legend(loc="upper right", fontsize=14)
        # plt.show()

        plt.savefig(f"{base_path}/{study_id}/{run_id}/negative_model.svg")

    @classmethod
    def nll_plot_train_test_curve_with_variance(cls, model, X, y, study_id, run_id, base_path, num_samples=10):
        plt.clf()

        if len(X.shape) == 3:
            X = X.reshape(-1, X.shape[2])

        train_model = BayesianRegression(
            alpha=model.alpha,
            beta=model.beta
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
    # 可视化：负对数似然曲线 (NLL) 带方差区间
    def plot_nll_curve_with_variance(cls, model, X, y, num_samples=10, type="test"):
        X = X.reshape(-1, X.shape[2])

        data_sizes = np.arange(10, len(X), 10)
        nll_means = []
        nll_stds = []

        for size in tqdm(data_sizes, desc="draw nll curve"):
            nll_samples = []

            # 使用多次采样来计算均值和方差
            for _ in range(num_samples):
                # 随机选择样本子集来进行拟合和计算
                indices = np.random.choice(np.arange(size), size=size, replace=True)
                X_subset = X[:size][indices]
                if type == "test":
                    nll = model.log_likelihood(X_subset, X_subset)
                else:
                    model.train(X_subset)
                    nll = model.log_likelihood(X_subset, X_subset)
                nll_samples.append(nll)

            # 计算每个数据集大小下的NLL均值和标准差
            nll_means.append(np.mean(nll_samples))
            nll_stds.append(np.std(nll_samples))

        nll_means = np.array(nll_means)
        nll_stds = np.array(nll_stds)

        # 绘制NLL均值曲线和方差区间

        if type == "test":
            pass
        else:
            plt.clf()
            plt.figure(figsize=(8, 6))

            # 用于显示中文
            # plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

        if type == "test":
            plt.plot(data_sizes, nll_means, label="validate", color="orange")
            plt.fill_between(data_sizes, nll_means - nll_stds, nll_means + nll_stds, color="orange", alpha=0.2,
                             label="验证集")
        else:
            plt.plot(data_sizes, nll_means, label="train", color="red")
            plt.fill_between(data_sizes, nll_means - nll_stds, nll_means + nll_stds, color="red", alpha=0.2,
                             label="训练集")

        if type == "test":
            # plt.title("Negative Log Likelihood (NLL) with Variance vs Data Size")
            plt.xlabel("Data Size")
            plt.ylabel("Negative Log Likelihood")
            plt.legend(loc="upper right")
            plt.show()
        else:
            return plt

    @classmethod
    def new_draw_anomaly_plots(cls,
                               anomaly_threshold_list,
                               positive_all_deviation: np.ndarray,
                               negative_all_deviation: np.ndarray,
                               study_id,
                               run_id,
                               base_path
                               ):

        plt.clf()

        combined_data = np.vstack((np.array(anomaly_threshold_list).reshape(1, -1), np.vstack(positive_all_deviation),
                                   np.vstack(negative_all_deviation)))

        # 训练 PCA 模型，不限制主成分个数
        pca = PCA(n_components=None)
        pca.fit(combined_data)

        # 获取解释方差比
        explained_variance_ratio = pca.explained_variance_ratio_

        # 累积解释方差比
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)

        print(f"cumulative_explained_variance: {cumulative_explained_variance}")

        # # 绘制解释方差比与累计解释方差比的图
        # plt.figure(figsize=(8, 6))
        # plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.6,
        #         label='Individual explained variance')
        # plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid',
        #          label='Cumulative explained variance')
        #
        # plt.xlabel('Number of Principal Components')
        # plt.ylabel('Explained Variance Ratio')
        # plt.legend(loc='best')
        # plt.tight_layout()
        # plt.show()
        #
        # # 打印出选择95%方差阈值下的最佳 k 值
        # k = np.argmax(cumulative_explained_variance >= 0.95) + 1
        # print(f"Best k (for 95% explained variance): {k}")

        component_num = 3

        pca = PCA(n_components=component_num)
        pca_result = pca.fit_transform(combined_data)

        pca_anomaly_threshold_list = pca_result[:len(np.array(anomaly_threshold_list).reshape(1, -1)), :]
        pca_positive_all_deviation = pca_result[len(np.array(anomaly_threshold_list).reshape(1, -1)): len(
            np.array(anomaly_threshold_list).reshape(1, -1)) + len(np.vstack(positive_all_deviation)), :]
        pca_negative_all_deviation = pca_result[len(np.array(anomaly_threshold_list).reshape(1, -1)) + len(
            np.vstack(positive_all_deviation)):, :]

        fig, axes = plt.subplots(1, component_num, figsize=(20, 7), dpi=900)

        for component in range(component_num):
            axes[component].scatter(
                [i for i in range(len(pca_negative_all_deviation))],
                pca_negative_all_deviation[:, component],
                s=2,
                color=MyColor.COLOR_1,
                label="negative deviation"
            )

            axes[component].scatter(
                [i for i in range(len(pca_positive_all_deviation))],
                pca_positive_all_deviation[:, component],
                s=2,
                color=MyColor.COLOR_3,
                label="positive deviation"
            )

            axes[component].axhline(pca_anomaly_threshold_list.reshape(-1)[component], c=MyColor.COLOR_4)

            # axes[component].set_title(f"{name} anomaly display")
            axes[component].set_xlabel("Num", fontsize=14, labelpad=20)
            axes[component].set_ylabel("Value", fontsize=14, labelpad=20)
            axes[component].legend(loc="upper right", fontsize=14)

        plt.tight_layout()
        # plt.show()

        plt.savefig(f"{base_path}/{study_id}/{run_id}/threshold.svg")

    @classmethod
    def draw_anomaly_plots(cls,
                           name: str,
                           anomaly_threshold: float,
                           all_deviation: np.ndarray,
                           all_positive_deviation: np.ndarray,
                           all_negative_deviation: np.ndarray,
                           base_path
                           ):

        plt.clf()

        plt.figure(figsize=(10, 6), dpi=300)
        # 用于显示中文
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.scatter(
            [i for i in range(len(all_deviation))],
            [x for x in all_deviation],
            s=1,
            alpha=0.7,
            color="#ffba00",
            label="all deviation"
        )

        plt.scatter(
            [i for i in range(len(all_positive_deviation))],
            [x for x in all_positive_deviation],
            s=1,
            alpha=0.7,
            color="#ff0000",
            label="all positive deviation"
        )

        plt.scatter(
            [i for i in range(len(all_negative_deviation))],
            [x for x in all_negative_deviation],
            s=1,
            alpha=0.7,
            color="#ff6300",
            label="all negative deviation"
        )

        plt.axhline(anomaly_threshold)

        plt.title(f"{name} anomaly display")
        plt.xlabel("num")
        plt.ylabel("value")
        plt.legend(loc="best")
        # plt.show()

        plt.savefig(f"{base_path}/{name}_anomaly_display.svg")
