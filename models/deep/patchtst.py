import numpy as np
import pandas as pd
import torch

from transformers import PatchTSTConfig, PatchTSTForPrediction

from ..base import ForecastModel


class PatchTSTModel(ForecastModel):

    def __init__(
        self,
        input_length=24,
        horizon=12,
        patch_length=4,
        patch_stride=2,
        d_model=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        epochs=50,
        batch_size=16,
        learning_rate=1e-3,
    ):

        self.name = "PatchTST"

        self.input_length = input_length
        self.horizon = horizon

        self.patch_length = patch_length
        self.patch_stride = patch_stride

        self.d_model = d_model
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = None
        self.history = None


    def _make_supervised(self, y):

        X = []
        target = []

        for t in range(
            self.input_length,
            len(y) - self.horizon + 1
        ):

            X.append(
                y[t - self.input_length:t]
            )

            target.append(
                y[t:t + self.horizon]
            )

        return (
            np.asarray(X),
            np.asarray(target),
        )


    def fit(self, y):

        y = (
            pd.Series(y)
            .dropna()
            .astype(float)
            .reset_index(drop=True)
            .values
        )

        if len(y) <= self.input_length + self.horizon:
            raise ValueError(
                "Training series is too short for PatchTST."
            )

        self.history = y.copy()

        X, target = self._make_supervised(y)

        # PatchTST expects:
        # (batch, sequence_length, channels)

        X = torch.tensor(
            X,
            dtype=torch.float32
        ).unsqueeze(-1)

        target = torch.tensor(
            target,
            dtype=torch.float32
        ).unsqueeze(-1)

        config = PatchTSTConfig(
            num_input_channels=1,
            context_length=self.input_length,
            prediction_length=self.horizon,

            patch_length=self.patch_length,
            patch_stride=self.patch_stride,

            d_model=self.d_model,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,

            loss="mse",
        )

        self.model = PatchTSTForPrediction(
            config
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        torch.set_num_threads(1)
        self.model.train()

        n_rows = X.shape[0]

        for _ in range(self.epochs):
            order = torch.randperm(n_rows)

            for start in range(0, n_rows, self.batch_size):
                index = order[start:start + self.batch_size]

                optimizer.zero_grad()

                outputs = self.model(
                    past_values=X[index],
                    future_values=target[index],
                )

                loss = outputs.loss

                loss.backward()

                optimizer.step()

        return self


    def predict(self, horizon):

        if horizon != self.horizon:
            raise ValueError(
                f"PatchTST currently expects horizon={self.horizon}."
            )

        x = self.history[
            -self.input_length:
        ]

        x = torch.tensor(
            x,
            dtype=torch.float32
        ).reshape(
            1,
            self.input_length,
            1,
        )

        self.model.eval()

        with torch.no_grad():

            outputs = self.model(
                past_values=x
            )

            forecast = (
                outputs
                .prediction_outputs
                .squeeze(0)
                .squeeze(-1)
                .cpu()
                .numpy()
            )

        return pd.Series(
            forecast
        )