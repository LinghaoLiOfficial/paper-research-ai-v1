import os
import importlib


def init_model_pattern_factory():

    # 动态扫描并加载所有模型训练测试模式类

    folder_path = "./pattern/data_analysis"
    model_pattern_mapping = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".py"):
            class_name = file_name.replace(".py", "")
            try:
                # 动态导入模块
                module = importlib.import_module(f'pattern.data_analysis.{class_name}')
                # 获取同名的类对象
                pattern_class = getattr(module, class_name)
                model_pattern_mapping[class_name] = pattern_class

            except Exception as e:
                print(e)
                continue

    return model_pattern_mapping


class ModelPatternFactory:

    model_pattern_mapping = init_model_pattern_factory()

    @classmethod
    def get_model_pattern(cls, model_name):
        return cls.model_pattern_mapping.get(model_name, None)

    @classmethod
    def get_all_model_pattern_name(cls):
        return [x for x in cls.model_pattern_mapping.keys()]
