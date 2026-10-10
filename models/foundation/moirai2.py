import numpy as np
import pandas as pd

from gluonts.dataset.common import ListDataset

from uni2ts.model.moirai2 import (
    Moirai2Forecast,
    Moirai2Module,
)

from ..base import ForecastModel


class Moirai2Model(ForecastModel):

    def __init__(
        self,
        model_id="Salesforce/moirai-2.0-R-small",
        freq="D",
        context_length=None,
        batch_size=1,
        return_quantiles=False,
        quantile_levels=None,
    ):

        self.name = "Moirai2"

        self.model_id = model_id
        self.freq = freq

        self.context_length = context_length
        self.batch_size = batch_size

        self.return_quantiles = return_quantiles

        self.quantile_levels = (
            quantile_levels
            if quantile_levels is not None
            else [0.1, 0.5, 0.9]
        )

        self.module = None
        self.target = None


    def _load_model(self):

        if self.module is None:

            self.module = (
                Moirai2Module.from_pretrained(
                    self.model_id
                )
            )


    def fit(self, y):

        # -------------------------
        # Convert input
        # -------------------------

        if isinstance(y, pd.Series):

            target = y.to_numpy(
                dtype=np.float32
            )

        elif isinstance(y, pd.DataFrame):

            # DataFrame:
            # rows = time
            # columns = variables
            #
            # Moirai expects:
            # (target_dim, time)

            target = (
                y
                .to_numpy(
                    dtype=np.float32
                )
                .T
            )

        else:

            target = np.asarray(
                y,
                dtype=np.float32
            )


        # -------------------------
        # Shape handling
        # -------------------------

        if target.ndim == 1:

            # univariate:
            # (time,)

            self.target = target

        elif target.ndim == 2:

            # multivariate:
            # (n_variates, time)

            self.target = target

        else:

            raise ValueError(
                "Input must be a 1D univariate series "
                "or a 2D multivariate array/DataFrame."
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


        # -------------------------
        # Determine dimensionality
        # -------------------------

        if self.target.ndim == 1:

            target_dim = 1
            history_length = len(
                self.target
            )

        else:

            target_dim = (
                self.target.shape[0]
            )

            history_length = (
                self.target.shape[1]
            )


        context_length = (
            self.context_length
            if self.context_length is not None
            else history_length
        )


        # -------------------------
        # GluonTS dataset
        # -------------------------

        dataset = ListDataset(
            [
                {
                    "start": pd.Period(
                        "2000-01-01",
                        freq=self.freq,
                    ),
                    "target": self.target,
                }
            ],
            freq=self.freq,
            one_dim_target=(
                target_dim == 1
            ),
        )


        # -------------------------
        # Joint Moirai2 model
        # -------------------------

        model = Moirai2Forecast(
            module=self.module,

            prediction_length=horizon,
            context_length=context_length,

            target_dim=target_dim,

            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )


        predictor = (
            model.create_predictor(
                batch_size=self.batch_size,
            )
        )


        forecast = next(
            predictor.predict(
                dataset
            )
        )


        # -------------------------
        # Point forecast = median
        # -------------------------

        point = np.asarray(
            forecast.quantile(0.5)
        )


        # GluonTS may return:
        # univariate: (horizon,)
        # multivariate: (horizon, target_dim)
        #
        # we want:
        # multivariate: (target_dim, horizon)

        if (
            target_dim > 1
            and point.ndim == 2
            and point.shape[0] == horizon
        ):

            point = point.T


        # -------------------------
        # Quantiles
        # -------------------------

        if return_quantiles:

            all_quantiles = []

            for q in self.quantile_levels:

                q_forecast = np.asarray(
                    forecast.quantile(q)
                )

                if (
                    target_dim > 1
                    and q_forecast.ndim == 2
                    and q_forecast.shape[0]
                    == horizon
                ):

                    q_forecast = (
                        q_forecast.T
                    )

                all_quantiles.append(
                    q_forecast
                )


            # univariate:
            # (horizon, n_quantiles)
            #
            # multivariate:
            # (target_dim, horizon, n_quantiles)

            quantiles = np.stack(
                all_quantiles,
                axis=-1,
            )


            return {
                "point_forecast": point,
                "quantiles": quantiles,
                "quantile_levels":
                    self.quantile_levels,
            }


        return point