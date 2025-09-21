import numpy as np


class KernelBayesianRegression:
    def __init__(self, alpha, beta, kernel='rbf', length_scale=0.5, kernel_params=None):
        self.alpha = alpha
        self.beta = beta
        self.kernel = kernel
        self.length_scale = length_scale
        self.kernel_params = kernel_params or {}

        # 初始化核函数
        if kernel == 'rbf':
            self.kernel_func = self.rbf_kernel
        else:
            raise ValueError('不支持的核函数')

        # 存储训练数据和对偶变量
        self.X_train = None
        self.A = None  # 对偶变量矩阵
        self.A_coeff = None  # 对偶系数矩阵

    def rbf_kernel(self, X1, X2, length_scale=None):
        l = length_scale or self.length_scale

        # 向量化计算核矩阵
        X1_norm = np.sum(X1**2, axis=1, keepdims=True)
        X2_norm = np.sum(X2**2, axis=1)
        cross_term = np.dot(X1, X2.T)
        dist_sq = X1_norm - 2 * cross_term + X2_norm

        # 防止数值问题
        dist_sq = np.clip(dist_sq, 1e-12, None)
        K = np.exp(- dist_sq / (2 * l ** 2))

        return K

    def train(self, X):
        # 处理输入数据的维度
        if len(X.shape) == 3:
            n_samples, time_steps, n_features = X.shape
            X_reshaped = X.reshape(n_samples, -1)
        else:
            X_reshaped = X
            time_steps, n_features = None, X.shape[1]

        n_samples, n_features_total = X_reshaped.shape

        # 保存训练数据
        self.X_train = X_reshaped

        # 计算核矩阵
        K = self.compute_kernel(X_reshaped, X_reshaped)

        # 添加正则化项
        lambda_val = self.alpha / self.beta
        reg_term = lambda_val * np.eye(n_samples)
        K_reg = K + reg_term

        # 计算对偶变量矩阵
        self.A = np.linalg.pinv(K_reg)

        # 计算对偶系数矩阵
        self.A_coeff = self.A @ X_reshaped

        # 计算重构数据
        reconstructed_flat = K @ self.A_coeff

        # 恢复原始形状
        if len(X.shape) == 3:
            reconstructed = reconstructed_flat.reshape(n_samples, time_steps, n_features)
        else:
            reconstructed = reconstructed_flat

        return reconstructed

    def compute_kernel(self, X1, X2):
        # 计算两个矩阵之间的核函数
        if self.kernel == 'rbf':
            return self.rbf_kernel(X1, X2)
        else:
            raise ValueError('核函数未正确初始化')

    def test(self, X):
        # 处理输入数据的维度
        if len(X.shape) == 3:
            n_samples, time_steps, n_features = X.shape
            X_reshaped = X.reshape(n_samples, -1)
        else:
            X_reshaped = X
            time_steps, n_features = None, X.shape[1]

        n_samples, n_features_total = X_reshaped.shape

        # 计算核矩阵
        k_star = self.compute_kernel(X_reshaped, self.X_train)

        # 预测输出
        predicted_flat = k_star @ self.A_coeff

        # 恢复原始形状
        if len(X.shape) == 3:
            predicted = predicted_flat.reshape(n_samples, time_steps, n_features)
        else:
            predicted = predicted_flat

        return predicted

    def predict(self, X):
        # 处理输入数据的维度
        if len(X.shape) == 3:
            n_samples, time_steps, n_features = X.shape
            X_reshaped = X.reshape(n_samples, -1)
        else:
            X_reshaped = X
            time_steps, n_features = None, X.shape[1]

        n_samples, n_features_total = X_reshaped.shape

        # 计算核矩阵
        k_star = self.compute_kernel(X_reshaped, self.X_train)

        # 计算预测均值
        mean_pred_flat = k_star @ self.A_coeff

        # 计算自相关项
        k_star_star = self.compute_kernel(X_reshaped, X_reshaped)
        # if k_star_star == 0:
        #     k_star_star = k_star_star.reshape(1, 1)

        # 计算预测方差
        variance_terms = np.diag(k_star_star) - np.sum(k_star @ self.A * k_star, axis=1) + 1/ self.beta

        # 确保方差不为负
        variance_terms = np.maximum(variance_terms, 1e-8)

        # 扩展方差到每个输出维度
        n_outputs = self.A_coeff.shape[1]
        var_pred_flat = np.tile(variance_terms[:, None], (1, n_outputs))

        # 恢复原始形状
        if len(X.shape) == 3:
            mean_pred = mean_pred_flat.reshape(n_samples, time_steps, n_features)
            var_pred = var_pred_flat.reshape(n_samples, time_steps, n_features)
        else:
            mean_pred = mean_pred_flat
            var_pred = var_pred_flat

        return mean_pred, var_pred

    def log_likelihood(self, X, y):
        # 展平输入和目标
        original_shape = X.shape
        if len(X.shape) == 3:
            n_samples, time_steps, n_features = X.shape
            X_reshaped = X.reshape(n_samples, -1)
            y_reshaped = y.reshape(n_samples, -1)
        else:
            X_reshaped = X
            y_reshaped = y

        # 获取预测结果
        mean_pred, var_pred = self.predict(X_reshaped)

        # 计算残差
        residual = y_reshaped - mean_pred
        # 每个样本的方差（标量）
        sigma_sq_samples = var_pred[:, 0]  # 每个输出维度的方差相同
        n_outputs = mean_pred.shape[1]

        # 向量化计算对数似然
        const_term = -0.5 * n_outputs * np.log(2 * np.pi)
        log_var = np.log(sigma_sq_samples)
        residual_ss = np.sum(residual ** 2, axis=1)
        log_likelihoods = const_term - 0.5 * n_outputs * log_var - 0.5 * residual_ss / sigma_sq_samples

        # 对数似然总和
        ll_total = np.sum(log_likelihoods)
        nll_total = -ll_total

        # 返回平均负对数似然
        return nll_total / X_reshaped.shape[0]

