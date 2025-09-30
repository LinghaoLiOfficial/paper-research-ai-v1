from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=120, random_state=42)

    def train(self, X, y):
        self.model = RandomForestClassifier(n_estimators=120, random_state=42)

        model = self.model
        model.fit(X, y)
        self.model = model

        pred = self.model.predict(X)
        return pred

    def test(self, X):
        pred = self.model.predict(X)
        return pred

