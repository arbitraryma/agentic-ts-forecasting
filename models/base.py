class ForecastModel:

    def fit(self, y):
        raise NotImplementedError

    def predict(self, horizon):
        raise NotImplementedError
