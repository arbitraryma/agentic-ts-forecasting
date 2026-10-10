import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from ..base import ForecastModel


class ETSModel(ForecastModel):

    def __init__(
        self,
        seasonal_periods=None,
        trend="add",
        seasonal="add",
    ):
        self.name = "ETS"

        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal

        self.model = None
        self.fitted_model = None


    def fit(self, y):

        y = (
            pd.Series(y)
            .dropna()
            .astype(float)
            .reset_index(drop=True)
        )

        if self.seasonal_periods is None:
            seasonal = None
        else:
            seasonal = self.seasonal

        self.model = ExponentialSmoothing(
            y,
            trend=self.trend,
            seasonal=seasonal,
            seasonal_periods=self.seasonal_periods,
        )

        self.fitted_model = self.model.fit()

        return self


    def predict(self, horizon):

        forecast = self.fitted_model.forecast(
            horizon
        )

        return pd.Series(forecast)