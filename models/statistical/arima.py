import pandas as pd
import pmdarima as pm

from ..base import ForecastModel


class AutoArima(ForecastModel):

    def __init__(self, seasonal=True, m=1):
        self.name = "AutoArima"
        self.seasonal = seasonal
        self.m = m
        self.model = None

    def fit(self, y):
        y = pd.Series(y).astype(float)

        self.model = pm.auto_arima(
            y,
            seasonal=self.seasonal,
            m=self.m,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )

        return self

    def predict(self, horizon):
        forecast = self.model.predict(
            n_periods=horizon
        )

        return pd.Series(forecast)