import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from ..base import ForecastModel


class XGBoostModel(ForecastModel):

    def __init__(
        self,
        lags=12,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
    ):
        self.name = "XGBoost"

        self.lags = lags

        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="reg:squarederror",
            random_state=42,
        )

        self.history = None


    def _make_supervised(self, y):

        X = []
        target = []

        y = np.asarray(
            y,
            dtype=float
        )

        for t in range(
            self.lags,
            len(y)
        ):

            X.append(
                y[
                    t - self.lags:t
                ]
            )

            target.append(
                y[t]
            )

        return (
            np.asarray(X),
            np.asarray(target)
        )


    def fit(self, y):

        y = (
            pd.Series(y)
            .dropna()
            .astype(float)
            .reset_index(drop=True)
        )

        if len(y) <= self.lags:
            raise ValueError(
                "Training series must be longer than lags."
            )

        self.history = y.tolist()

        X, target = self._make_supervised(
            y
        )

        self.model.fit(
            X,
            target
        )

        return self


    def predict(self, horizon):

        history = self.history.copy()

        forecasts = []

        for _ in range(horizon):

            x = np.array(
                history[-self.lags:]
            ).reshape(1, -1)

            prediction = float(
                self.model.predict(x)[0]
            )

            forecasts.append(
                prediction
            )

            history.append(
                prediction
            )

        return pd.Series(
            forecasts
        )