import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RegressionIndicatorsParser:

    ROUND_RESERVE = 8

    def __init__(self):

        self.loss = 0
        self.mse = 0
        self.rmse = 0
        self.mae = 0
        self.r2 = 0
        self.adjusted_r2 = 0
        self.mape = 0

        self.loss_num = 0
        self.mse_num = 0
        self.rmse_num = 0
        self.mae_num = 0
        self.r2_num = 0
        self.adjusted_r2_num = 0
        self.mape_num = 0

        self.other = None

        self.eps = 1e-8

    def display_result(self) -> dict:

        avg_loss = round(self.loss / self.loss_num, self.ROUND_RESERVE) if self.loss_num != 0 else 0.0

        avg_mse = round(self.mse / self.mse_num, self.ROUND_RESERVE) if self.mse_num != 0 else 0.0

        avg_rmse = round(self.rmse / self.rmse_num, self.ROUND_RESERVE) if self.rmse_num != 0 else 0.0

        avg_mae = round(self.mae / self.mae_num, self.ROUND_RESERVE) if self.mae_num != 0 else 0.0

        avg_r2 = round(self.r2 / self.r2_num, self.ROUND_RESERVE) if self.r2_num != 0 else 0.0

        avg_adjusted_r2 = round(self.adjusted_r2 / self.adjusted_r2_num, self.ROUND_RESERVE) if self.adjusted_r2_num != 0 else 0.0

        avg_mape = round(self.mape / self.mape_num, self.ROUND_RESERVE) if self.mape_num != 0 else 0.0

        result_dict = {
            "loss": round(avg_loss, 4),
            "mse": round(avg_mse, 4),
            "rmse": round(avg_rmse, 4),
            "mae": round(avg_mae, 4),
            "r2": round(avg_r2, 4),
            "adjusted_r2": round(avg_adjusted_r2, 4),
            "mape": round(avg_mape, 4),
        }

        return result_dict

    def calculate_loss(self, loss) -> float:

        self.loss += loss
        self.loss_num += 1

        return loss

    def calculate_mse(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:

        if len(real_array.shape) == 3:
            real_array = real_array.reshape(real_array.shape[0], real_array.shape[1] * real_array.shape[2])

        if len(pred_array.shape) == 3:
            pred_array = pred_array.reshape(pred_array.shape[0], pred_array.shape[1] * pred_array.shape[2])

        mse = mean_squared_error(
            y_true=real_array,
            y_pred=pred_array
        )

        self.mse += mse
        self.mse_num += 1

        return mse

    def calculate_rmse(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:

        if len(real_array.shape) == 3:
            real_array = real_array.reshape(real_array.shape[0], real_array.shape[1] * real_array.shape[2])

        if len(pred_array.shape) == 3:
            pred_array = pred_array.reshape(pred_array.shape[0], pred_array.shape[1] * pred_array.shape[2])

        mse = self.calculate_mse(
            real_array=real_array,
            pred_array=pred_array
        )

        rmse = np.sqrt(mse)

        self.rmse += rmse
        self.rmse_num += 1

        return rmse

    def calculate_mae(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:

        if len(real_array.shape) == 3:
            real_array = real_array.reshape(real_array.shape[0], real_array.shape[1] * real_array.shape[2])

        if len(pred_array.shape) == 3:
            pred_array = pred_array.reshape(pred_array.shape[0], pred_array.shape[1] * pred_array.shape[2])

        mae = mean_absolute_error(
            y_true=real_array,
            y_pred=pred_array
        )

        self.mae += mae
        self.mae_num += 1

        return mae

    def calculate_r2(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:

        if len(real_array.shape) == 3:
            real_array = real_array.reshape(real_array.shape[0], real_array.shape[1] * real_array.shape[2])

        if len(pred_array.shape) == 3:
            pred_array = pred_array.reshape(pred_array.shape[0], pred_array.shape[1] * pred_array.shape[2])

        y_mean = np.mean(real_array)
        denominator = np.sum((real_array - y_mean) ** 2)
        if denominator < self.eps:  # 分母接近0时返回0
            return 0.0
        r2 = 1 - (np.sum((real_array - pred_array) ** 2) / denominator)

        if not np.isnan(r2):
            self.r2 += r2
            self.r2_num += 1

        return r2

    def calculate_adjusted_r2(self, real_array: np.ndarray, pred_array: np.ndarray, p) -> float:

        # 样本量不足以计算调整R²
        if len(real_array) <= p + 1:
            return 0

        if len(real_array.shape) == 3:
            real_array = real_array.reshape(real_array.shape[0], real_array.shape[1] * real_array.shape[2])

        if len(pred_array.shape) == 3:
            pred_array = pred_array.reshape(pred_array.shape[0], pred_array.shape[1] * pred_array.shape[2])

        r2 = r2_score(
            y_true=real_array,
            y_pred=pred_array
        )

        adjusted_r2 = 1 - ((1 - r2) * (len(real_array) - 1)) / max(len(real_array) - p - 1, self.eps)

        if not np.isnan(adjusted_r2):
            self.adjusted_r2 += adjusted_r2
            self.adjusted_r2_num += 1

        return adjusted_r2

    def calculate_mape(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:

        if len(real_array.shape) == 3:
            real_array = real_array.reshape(real_array.shape[0], real_array.shape[1] * real_array.shape[2])

        if len(pred_array.shape) == 3:
            pred_array = pred_array.reshape(pred_array.shape[0], pred_array.shape[1] * pred_array.shape[2])

        real_array = np.where(np.abs(real_array) < self.eps, self.eps, real_array)  # 替换接近0的值
        mape = np.mean(np.abs((real_array - pred_array) / real_array)) * 100

        if not np.isnan(mape):
            self.mape += mape
            self.mape_num += 1

        return mape