import numpy as np
import pandas as pd

from chronos import Chronos2Pipeline

from ..base import ForecastModel


class Chronos2Model(ForecastModel):

    def __init__(
        self,
        model_id="amazon/chronos-2",
        device="cpu",
        quantile_levels=None,
    ):
        self.name = "Chronos2"

        self.model_id = model_id
        self.device = device

        self.quantile_levels = (
            quantile_levels
            if quantile_levels is not None
            else [0.1, 0.5, 0.9]
        )

        self.pipeline = None

        self.target = None
        self.past_covariates = None
        self.future_covariates = None


    def _load_pipeline(self):

        if self.pipeline is None:
            self.pipeline = Chronos2Pipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
            )


    def fit(
        self,
        y,
        past_covariates=None,
        future_covariates=None,
    ):

        # -------------------------
        # Target
        # -------------------------

        if isinstance(y, pd.Series):
            y = y.to_numpy()

        elif isinstance(y, pd.DataFrame):
            # rows = time
            # columns = variables
            y = y.to_numpy().T

        else:
            y = np.asarray(y)

        if y.ndim == 1:
            # univariate
            # shape: (history_length,)
            pass

        elif y.ndim == 2:
            # multivariate
            # shape: (n_variates, history_length)
            pass

        else:
            raise ValueError(
                "y must be a 1D univariate series "
                "or 2D multivariate array/DataFrame."
            )

        self.target = y.astype(float)


        # -------------------------
        # Past covariates
        # -------------------------

        if past_covariates is not None:

            if isinstance(past_covariates, pd.DataFrame):
                past_covariates = {
                    col: past_covariates[col].to_numpy()
                    for col in past_covariates.columns
                }

            elif not isinstance(past_covariates, dict):
                raise ValueError(
                    "past_covariates must be a dict "
                    "or pandas DataFrame."
                )

            self.past_covariates = {
                key: np.asarray(value)
                for key, value in past_covariates.items()
            }

        else:
            self.past_covariates = None


        # -------------------------
        # Future covariates
        # -------------------------

        if future_covariates is not None:

            if isinstance(future_covariates, pd.DataFrame):
                future_covariates = {
                    col: future_covariates[col].to_numpy()
                    for col in future_covariates.columns
                }

            elif not isinstance(future_covariates, dict):
                raise ValueError(
                    "future_covariates must be a dict "
                    "or pandas DataFrame."
                )

            self.future_covariates = {
                key: np.asarray(value)
                for key, value in future_covariates.items()
            }

        else:
            self.future_covariates = None


        # -------------------------
        # Validation
        # -------------------------

        if self.past_covariates is not None:

            history_length = self.target.shape[-1]

            for key, value in self.past_covariates.items():

                if len(value) != history_length:
                    raise ValueError(
                        f"Past covariate '{key}' must have "
                        f"length {history_length}."
                    )


        if self.future_covariates is not None:

            if self.past_covariates is None:
                raise ValueError(
                    "future_covariates require "
                    "past_covariates."
                )

            missing_keys = (
                set(self.future_covariates.keys())
                - set(self.past_covariates.keys())
            )

            if missing_keys:
                raise ValueError(
                    "Every future covariate must also "
                    "appear in past_covariates. "
                    f"Missing: {missing_keys}"
                )


        self._load_pipeline()

        return self


    def predict(
        self,
        horizon,
        return_quantiles=False,
    ):

        # -------------------------
        # Check future covariates
        # -------------------------

        if self.future_covariates is not None:

            for key, value in self.future_covariates.items():

                if len(value) != horizon:
                    raise ValueError(
                        f"Future covariate '{key}' must have "
                        f"length equal to horizon={horizon}."
                    )


        # -------------------------
        # Build Chronos-2 input
        # -------------------------

        input_item = {
            "target": self.target
        }

        if self.past_covariates is not None:
            input_item[
                "past_covariates"
            ] = self.past_covariates

        if self.future_covariates is not None:
            input_item[
                "future_covariates"
            ] = self.future_covariates


        inputs = [input_item]


        # -------------------------
        # Forecast
        # -------------------------

        quantiles, point_forecast = (
            self.pipeline.predict_quantiles(
                inputs=inputs,
                prediction_length=horizon,
                quantile_levels=self.quantile_levels,
            )
        )


        # one forecasting task
        point = (
            point_forecast[0]
            .cpu()
            .numpy()
        )
        if point.ndim == 2 and point.shape[0] == 1:
            point = point[0]
            
        quantiles = (
            quantiles[0]
            .cpu()
            .numpy()
        )


        if return_quantiles:

            return {
                "point_forecast": point,
                "quantiles": quantiles,
                "quantile_levels": self.quantile_levels,
            }


        return point