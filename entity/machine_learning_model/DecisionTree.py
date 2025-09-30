from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(self):
        self.model = DecisionTreeClassifier(criterion='entropy', random_state=42)

    def train(self, X, y):
        model = self.model
        model.fit(X, y)
        self.model = model

        pred = self.model.predict(X)
        return pred

    def test(self, X):
        pred = self.model.predict(X)
        return pred

