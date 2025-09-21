import json

import torch
from datetime import datetime


class BestModelParser:

    def __init__(self, user_id, task_id):

        self.model_save_path = f"./storage/{user_id}/data_analysis/{task_id}"

        self.best_indicators_result = {}
        self.best_params = {}

    def display_result(self) -> dict:

        return self.best_indicators_result

    def compare_smaller_save(self, indicator: str, indicators_result: dict, model, epoch, params, history_id, study_id, run_id):

        if indicator not in self.best_indicators_result.keys() or \
                indicators_result[indicator] < self.best_indicators_result[indicator]:
            self.best_params = params

            indicators_result["epoch"] = epoch
            self.best_indicators_result = indicators_result

            self.best_indicators_result["history_id"] = history_id

            torch.save(model.state_dict(), f"{self.model_save_path}/{study_id}/{run_id}/model.pth")

    def compare_bigger_save(self, indicator: str, indicators_result: dict, model, epoch, params, history_id, study_id, run_id):

        if indicator not in self.best_indicators_result.keys() or \
                indicators_result[indicator] > self.best_indicators_result[indicator]:

            self.best_params = params

            indicators_result["epoch"] = epoch
            self.best_indicators_result = indicators_result

            self.best_indicators_result["history_id"] = history_id

            torch.save(model.state_dict(), f"{self.model_save_path}/{study_id}/{run_id}/model.pth")


