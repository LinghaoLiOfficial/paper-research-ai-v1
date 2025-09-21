from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


class SVM:
    def __init__(self):
        self.model = SVC(kernel="rbf")

    def train(self, X, y):
        X = X.reshape(X.shape[0], -1)
        y = y.reshape(y.shape[0], -1).ravel()

        model = self.model
        model.fit(X, y)
        self.model = model

        pred = self.model.predict(X)
        return pred

    def test(self, X):
        X = X.reshape(X.shape[0], -1)

        pred = self.model.predict(X)
        return pred

