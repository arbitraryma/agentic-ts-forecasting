import pandas as pd

from ..base import ForecastModel


class SeasonalNaive(ForecastModel):

    def __init__(self, season_length=1):
        self.season_length = season_length
        self.history = None
        self.name = 'SeasonalNaive'

    def fit(self, y):
        self.history = pd.Series(y).reset_index(drop=True)

        if len(self.history) < self.season_length:
            raise ValueError(
                "Training series is shorter than season_length."
            )

        return self

    def predict(self, horizon):
        last_season = self.history.iloc[-self.season_length:]

        forecasts = []

        for i in range(horizon):
            value = last_season.iloc[i % self.season_length]
            forecasts.append(value)

        return pd.Series(forecasts)