import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ..base import ForecastModel

class DLinearNet(nn.Module):
    def __init__(
        self,
        input_length,
        horizon,
        moving_avg=25,
    ):
        super().__init__()

        self.input_length = input_length
        self.horizon = horizon

        self.moving_avg = nn.AvgPool1d(
            kernel_size=moving_avg,
            stride=1,
            padding=moving_avg // 2,
        )

        self.linear_trend = nn.Linear(
            input_length,
            horizon,
        )

        self.linear_seasonal = nn.Linear(
            input_length,
            horizon,
        )

    def forward(self, x):

        x_3d = x.unsqueeze(1)

        trend = self.moving_avg(x_3d).squeeze(1)

        if trend.shape[1] > x.shape[1]:
            trend = trend[:, :x.shape[1]]

        seasonal = x - trend

        trend_out = self.linear_trend(trend)
        seasonal_out = self.linear_seasonal(seasonal)

        return trend_out + seasonal_out


class DLinearModel(ForecastModel):

    def __init__(
        self,
        input_length=24,
        horizon=12,
        moving_avg=25,
        epochs=50,
        batch_size=16,
        learning_rate=1e-3,
    ):
        self.name = "DLinear"

        self.input_length = input_length
        self.horizon = horizon
        self.moving_avg = moving_avg

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.net = None
        self.history = None

    def fit(self, y):
        y = (
            pd.Series(y)
            .dropna()
            .astype(float)
            .reset_index(drop=True)
        )

        self.history = y

        values = y.to_numpy(dtype=np.float32)

        if len(values) < self.input_length + self.horizon:
            raise ValueError(
                "Training series is shorter than input_length + horizon."
            )

        windows = []
        targets = []

        last_start = len(values) - self.input_length - self.horizon + 1

        for start in range(last_start):
            end = start + self.input_length
            windows.append(values[start:end])
            targets.append(values[end:end + self.horizon])

        x = torch.tensor(np.stack(windows))
        target = torch.tensor(np.stack(targets))

        self.net = DLinearNet(
            self.input_length,
            self.horizon,
            self.moving_avg,
        )

        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
        )

        loss_fn = nn.MSELoss()

        torch.set_num_threads(1)
        self.net.train()

        n_rows = x.shape[0]

        for _ in range(self.epochs):
            order = torch.randperm(n_rows)

            for start in range(0, n_rows, self.batch_size):
                index = order[start:start + self.batch_size]
                batch_x = x[index]
                batch_y = target[index]

                optimizer.zero_grad()
                loss = loss_fn(self.net(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, horizon):
        if self.net is None:
            raise ValueError("Call fit before predict.")

        if horizon != self.horizon:
            raise ValueError(
                f"DLinear currently expects horizon={self.horizon}."
            )

        window = (
            self.history
            .iloc[-self.input_length:]
            .to_numpy(dtype=np.float32)
        )

        self.net.eval()

        with torch.no_grad():
            forecast = self.net(
                torch.tensor(window).unsqueeze(0)
            )

        return pd.Series(
            forecast.squeeze(0).numpy()
        )