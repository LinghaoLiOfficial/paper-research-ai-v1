from sklearn.linear_model import LogisticRegression


class Logistic:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X, y):
        self.model = LogisticRegression()

        model = self.model
        model.fit(X, y)
        self.model = model

        pred = self.model.predict(X)
        return pred

    def test(self, X):
        pred = self.model.predict(X)
        return pred

