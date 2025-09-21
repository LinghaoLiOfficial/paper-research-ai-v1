import json
import os

import optuna


class HyperParamsParser:
    @classmethod
    def get_key(cls, config: dict):
        return [x for x in config.keys()][0]

    @classmethod
    def get_value(cls, config: dict):
        return [x for x in config.values()][0]
    
    @classmethod
    def parse(cls, param_dict: dict, trail: optuna.trial):
        new_param_dict = {}
        for param_name, param_config in param_dict.items():
            if isinstance(param_config, dict) and 'type' in param_config.keys():
                if param_config['type'] == "int":
                    new_param_dict[param_name] = trail.suggest_int(param_name, param_config['start'], param_config['end'])
                elif param_config['type'] == "float":
                    new_param_dict[param_name] = trail.suggest_float(param_name, param_config['start'], param_config['end'])
            else:
                new_param_dict[param_name] = param_config

        return new_param_dict

    @classmethod
    def load(cls, model_name: str = None) -> dict:

        root_path = "./config/hyper_params/files"
        json_path_list = [x for x in os.listdir(root_path)]

        hyper_params = {}
        for path in json_path_list:

            if "base" in path or model_name in path:

                with open("{}/{}".format(root_path, path), "r", encoding="utf-8") as f:

                    current_hyper_params = json.load(f)

                    hyper_params.update(current_hyper_params)

        return hyper_params
