class AnlpModel:
    def fit(self, train_data, train_labels):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError