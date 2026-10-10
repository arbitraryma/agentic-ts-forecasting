import numpy as np
import pandas as pd

from timesfm3 import TimesFM3Evaluator, ModelConfig

from ..base import ForecastModel


class TimesFM3Model(ForecastModel):

    def __init__(
        self,
        checkpoint_path="google/timesfm-3.0-pytorch",
        device="cpu",
        per_core_batch_size=4,
        return_quantiles=False,
    ):
        self.name = "TimesFM3"

        self.checkpoint_path = checkpoint_path
        self.device = device
        self.per_core_batch_size = per_core_batch_size
        self.return_quantiles = return_quantiles

        self.forecaster = None

        self.target = None
        self.past_only_covariates = None
        self.past_future_covariates = None


    def _load_model(self):

        if self.forecaster is None:

            config = ModelConfig(
                checkpoint_path=self.checkpoint_path,
                per_core_batch_size=self.per_core_batch_size,
                device=self.device,
            )

            self.forecaster = TimesFM3Evaluator(
                config
            )


    def _to_2d(self, x):

        if x is None:
            return None

        if isinstance(x, pd.Series):
            x = x.to_numpy()

        elif isinstance(x, pd.DataFrame):
            # rows = time
            # columns = variables
            x = x.to_numpy().T

        else:
            x = np.asarray(x)

        x = x.astype(np.float32)

        if x.ndim == 1:
            x = x[None, :]

        if x.ndim != 2:
            raise ValueError(
                "Input must be 1D or 2D."
            )

        return x


    def fit(
        self,
        y,
        past_only_covariates=None,
        past_future_covariates=None,
    ):

        self.target = self._to_2d(y)

        self.past_only_covariates = (
            self._to_2d(
                past_only_covariates
            )
        )

        self.past_future_covariates = (
            self._to_2d(
                past_future_covariates
            )
        )


        context_length = (
            self.target.shape[1]
        )


        if self.past_only_covariates is not None:

            if (
                self.past_only_covariates.shape[1]
                != context_length
            ):
                raise ValueError(
                    "past_only_covariates must have "
                    "the same context length as target."
                )


        if self.past_future_covariates is not None:

            if (
                self.past_future_covariates.shape[1]
                <= context_length
            ):
                raise ValueError(
                    "past_future_covariates must contain "
                    "both historical and future values."
                )


        self._load_model()

        return self


    def predict(
        self,
        horizon,
        return_quantiles=None,
    ):

        if return_quantiles is None:
            return_quantiles = (
                self.return_quantiles
            )


        context_length = (
            self.target.shape[1]
        )


        if self.past_future_covariates is not None:

            expected_length = (
                context_length + horizon
            )

            if (
                self.past_future_covariates.shape[1]
                != expected_length
            ):
                raise ValueError(
                    "past_future_covariates must have "
                    f"length context+horizon={expected_length}."
                )


        outputs = list(
            self.forecaster.predict_batch(
                contexts=[
                    self.target
                ],
                horizon=horizon,

                past_only_covariates=(
                    [self.past_only_covariates]
                    if self.past_only_covariates
                    is not None
                    else None
                ),

                past_future_covariates=(
                    [self.past_future_covariates]
                    if self.past_future_covariates
                    is not None
                    else None
                ),

                return_quantiles=True,
                use_symmetric_averaging=False,
            )
        )


        result = outputs[0]

        point = np.asarray(
            result.forecast
        )

        quantiles = np.asarray(
            result.quantiles
        )


        # univariate:
        # (1, horizon) -> (horizon,)
        if (
            point.ndim == 2
            and point.shape[0] == 1
        ):
            point = point[0]

        if (
            quantiles.ndim == 3
            and quantiles.shape[0] == 1
        ):
            quantiles = quantiles[0]


        if return_quantiles:

            return {
                "point_forecast": point,
                "quantiles": quantiles,
            }


        return point