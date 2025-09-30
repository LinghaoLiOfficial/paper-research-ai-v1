import numpy as np


class BayesianRegression:
    def __init__(self, alpha, beta):
        self.alpha = alpha  # 先验分布精度
        self.beta = beta  # 后验分布精度
        self.m_N = None  # 后验均值
        self.S_N = None  # 后验协方差

    def train(self, x):

        if len(x.shape) == 3:
            x_reshaped = x.reshape(-1, x.shape[2])
        else:
            x_reshaped = x

        # 计算贝叶斯多元回归的后验分布
        X_transpose = x_reshaped.T
        I = np.eye(x_reshaped.shape[1])
        S0_inv = self.alpha * I

        # 计算后验分布的精度矩阵
        S_N_inv = S0_inv + self.beta * X_transpose @ x_reshaped
        self.S_N = np.linalg.inv(S_N_inv)

        # 计算后验分布的均值
        self.m_N = self.beta * self.S_N @ (X_transpose @ x_reshaped)

        # 估计重构矩阵
        reconstructed_x = x_reshaped @ self.m_N.T

        if len(x.shape) == 3:
            y = reconstructed_x.reshape(-1, x.shape[1], x.shape[2])
        else:
            y = reconstructed_x

        return y

    def test(self, x):
        if len(x.shape) == 3:
            x_reshaped = x.reshape(-1, x.shape[2])
        else:
            x_reshaped = x

        # 估计重构矩阵
        reconstructed_x = x_reshaped @ self.m_N.T

        if len(x.shape) == 3:
            y = reconstructed_x.reshape(-1, x.shape[1], x.shape[2])
        else:
            y = reconstructed_x

        return y

    """
        绘图
    """

    def predict(self, X):
        # X = X.reshape(-1, X.shape[-1])  # 确保 X 是二维的
        mean_y = X @ self.m_N.T  # 预测均值
        var_y = 1 / self.beta + np.sum(X @ self.S_N * X, axis=1, keepdims=True)  # 预测方差
        return mean_y, var_y

    def log_likelihood(self, X, y):
        """计算负对数似然 (Negative Log Likelihood, NLL)"""
        mean_y, var_y = self.predict(X)
        residual = y - mean_y
        ll = -0.5 * np.sum(np.log(2 * np.pi * var_y) + (residual ** 2) / var_y)
        return -ll / len(y)  # 返回负对数似然
