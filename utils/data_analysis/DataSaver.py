import pickle
import os


class DataSaver:

    @classmethod
    def load_train_test(cls, turbine_num: int):

        x_train = []
        for i in range(turbine_num):
            x_train.append(cls.load("x_train_{}".format(i)))

        y_train = []
        for i in range(turbine_num):
            y_train.append(cls.load("y_train_{}".format(i)))

        x_test = cls.load("x_test")
        y_test = cls.load("y_test")

        return x_train, y_train, x_test, y_test

    @classmethod
    def check(cls, data_name: str):

        return os.path.exists('./data/cache/{}.pkl'.format(data_name))

    @classmethod
    def save(cls, data_name: str, data):

        with open('./data/cache/{}.pkl'.format(data_name), 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, data_name: str):

        with open('./data/cache/{}.pkl'.format(data_name), 'rb') as f:
            data = pickle.load(f)

        return data
