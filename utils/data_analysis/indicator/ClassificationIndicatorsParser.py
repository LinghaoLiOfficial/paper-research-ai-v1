import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ClassificationIndicatorsParser:

    ROUND_RESERVE = 8

    def __init__(self):

        self.loss = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.balanced_accuracy_with_recall = 0

        self.loss_num = 0
        self.accuracy_num = 0
        self.precision_num = 0
        self.recall_num = 0
        self.f1_num = 0
        self.balanced_accuracy_with_recall_num = 0

        self.other = None

    def display_result(self) -> dict:

        avg_loss = round(self.loss / self.loss_num, self.ROUND_RESERVE) if self.loss_num != 0 else 0.0

        avg_accuracy = round(self.accuracy / self.accuracy_num, self.ROUND_RESERVE) if self.accuracy_num != 0 else 0.0

        avg_precision = round(self.precision / self.precision_num, self.ROUND_RESERVE) if self.precision_num != 0 else 0.0

        avg_recall = round(self.recall / self.recall_num, self.ROUND_RESERVE) if self.recall_num != 0 else 0.0

        avg_f1 = round(self.f1 / self.f1_num, self.ROUND_RESERVE) if self.f1_num != 0 else 0.0

        avg_balanced_accuracy_with_recall = round(self.balanced_accuracy_with_recall / self.balanced_accuracy_with_recall_num, self.ROUND_RESERVE) if self.balanced_accuracy_with_recall_num != 0 else 0.0

        result_dict = {
            "loss": round(avg_loss, 4),
            "accuracy": round(avg_accuracy, 4),
            "precision": round(avg_precision, 4),
            "recall": round(avg_recall, 4),
            "f1": round(avg_f1, 4),
            "balanced_accuracy_with_recall": round(avg_balanced_accuracy_with_recall, 4)
        }

        return result_dict

    def calculate_balanced_accuracy_with_recall(self, accuracy, recall) -> float:

        denominator = accuracy + recall
        if denominator == 0:
            # 当两者均为0时视为无效指标
            return 0.0 if (accuracy == 0 and recall == 0) else 2 * accuracy * recall / (
                        denominator + np.finfo(float).eps)
        return 2 * accuracy * recall / denominator

    def calculate_loss(self, loss) -> float:

        self.loss += loss
        self.loss_num += 1

        return loss

    def calculate_accuracy(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:

        accuracy = accuracy_score(
            y_true=real_array,
            y_pred=pred_array
        )

        self.accuracy += accuracy
        self.accuracy_num += 1

        return accuracy

    def calculate_precision(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:

        precision = precision_score(
            y_true=real_array,
            y_pred=pred_array,
            zero_division=np.nan
        )

        if not np.isnan(precision):
            self.precision += precision
            self.precision_num += 1

        return precision

    def calculate_recall(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:

        recall = recall_score(
            y_true=real_array,
            y_pred=pred_array,
            zero_division=np.nan
        )

        if not np.isnan(recall):
            self.recall += recall
            self.recall_num += 1

        return recall

    def calculate_f1(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:

        f1 = f1_score(
            y_true=real_array,
            y_pred=pred_array,
            zero_division=np.nan
        )

        if not np.isnan(f1):
            self.f1 += f1
            self.f1_num += 1

        return f1



