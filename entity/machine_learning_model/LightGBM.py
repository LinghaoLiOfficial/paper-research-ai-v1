import lightgbm as lgb


class LightGBM:
    def __init__(self):
        self.params = {
            'objective': 'binary',      # 二分类
            'metric': 'binary_logloss', # 使用binary_logloss作为评估指标
            'boosting_type': 'gbdt',    # 使用梯度提升决策树 (GBDT)
            'num_leaves': 31,           # 控制树的复杂度
            'learning_rate': 0.05,      # 学习率
            'feature_fraction': 0.9     # 每次迭代时使用90%的特征
        }

        self.model = None

    def train(self, x_train, y_train, x_test, y_test):

        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        train_dataset = lgb.Dataset(x_train, label=y_train)
        test_dataset = lgb.Dataset(x_test, label=y_test, reference=train_dataset)

        self.model = lgb.train(self.params, train_dataset, valid_sets=[test_dataset])

        y_train_pred_prob = self.model.predict(x_train, num_iteration=self.model.best_iteration)
        y_train_pred = (y_train_pred_prob > 0.5).astype(int)  # 将概率转化为二进制分类

        return y_train_pred

    def test(self, x):

        x = x.reshape(x.shape[0], -1)

        y_pred_prob = self.model.predict(x, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_prob > 0.5).astype(int)  # 将概率转化为二进制分类

        return y_pred
