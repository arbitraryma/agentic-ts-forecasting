# pip install neuralforecast
import pandas as pd 
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet
from ..base import ForecastModel

class TimesNetModel(ForecastModel):

    def __init__(
        self,
        input_length=24,
        horizon=12,
        top_k=3,
        hidden_size=32,
        encoder_layers=2,
        max_steps=100,
        freq="D",
    ):

        self.name = "TimesNet"

        self.input_length = input_length
        self.horizon = horizon

        self.top_k = top_k
        self.hidden_size = hidden_size
        self.encoder_layers = encoder_layers

        self.max_steps = max_steps
        self.freq = freq

        self.model = None
        self.nf = None
        self.history = None


    def fit(self, y):

        y = (
            pd.Series(y)
            .dropna()
            .astype(float)
            .reset_index(drop=True)
        )

        self.history = y.copy()

        df = pd.DataFrame({
            "unique_id": "series_1",
            "ds": pd.date_range(
                start="2000-01-01",
                periods=len(y),
                freq=self.freq,
            ),
            "y": y.values,
        })

        self.model = TimesNet(
            h=self.horizon,
            input_size=self.input_length,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            encoder_layers=self.encoder_layers,
            max_steps=self.max_steps,
        )

        self.nf = NeuralForecast(
            models=[self.model],
            freq=self.freq,
        )

        self.nf.fit(
            df=df
        )

        return self


    def predict(self, horizon):

        if horizon != self.horizon:
            raise ValueError(
                f"TimesNet currently expects horizon={self.horizon}."
            )

        forecast_df = self.nf.predict()

        forecast = forecast_df[
            "TimesNet"
        ].values

        return pd.Series(
            forecast
        )