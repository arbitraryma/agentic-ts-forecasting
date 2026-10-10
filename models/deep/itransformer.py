import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import iTransformer
from ..base import ForecastModel


class ITransformerModel(ForecastModel):

    def __init__(
        self,
        input_length=24,
        horizon=12,
        n_series=1,
        hidden_size=64,
        n_heads=4,
        e_layers=2,
        max_steps=100,
        learning_rate=1e-3,
        freq="D",
    ):

        self.name = "iTransformer"

        self.input_length = input_length
        self.horizon = horizon
        self.n_series = n_series

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.e_layers = e_layers

        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.freq = freq

        self.model = None
        self.nf = None


    def fit(self, y):

        y = (
            pd.Series(y)
            .dropna()
            .astype(float)
            .reset_index(drop=True)
        )

        df = pd.DataFrame({
            "unique_id": "series_1",
            "ds": pd.date_range(
                start="2000-01-01",
                periods=len(y),
                freq=self.freq,
            ),
            "y": y.values,
        })

        self.model = iTransformer(
            h=self.horizon,
            input_size=self.input_length,
            n_series=self.n_series,

            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            e_layers=self.e_layers,

            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
        )

        self.nf = NeuralForecast(
            models=[self.model],
            freq=self.freq,
        )

        self.nf.fit(df=df)

        return self


    def predict(self, horizon):

        if horizon != self.horizon:
            raise ValueError(
                f"iTransformer currently expects horizon={self.horizon}."
            )

        forecast_df = self.nf.predict()

        forecast = forecast_df[
            "iTransformer"
        ].values

        return pd.Series(forecast)