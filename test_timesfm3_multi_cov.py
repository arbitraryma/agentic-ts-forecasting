import numpy as np
import pandas as pd

from models.foundation.timesfm3 import TimesFM3Model


targets = pd.DataFrame({
    "load": [
        100, 102, 101, 105, 108,
        110, 111, 115, 117, 118,
        120, 122, 125, 127, 129,
        130, 133, 135, 137, 140
    ],

    "price": [
        50, 51, 50, 52, 53,
        54, 53, 55, 56, 57,
        58, 59, 60, 60, 61,
        62, 63, 64, 65, 66
    ]
})


past_only = pd.DataFrame({
    "promo": [
        0, 0, 0, 0, 0,
        1, 1, 0, 0, 0,
        0, 0, 1, 1, 0,
        0, 0, 0, 0, 0
    ]
})


past_future = pd.DataFrame({
    "temperature": [
        25, 25, 26, 26, 27,
        27, 28, 28, 29, 29,
        30, 30, 31, 31, 32,
        32, 33, 33, 34, 34,

        # future 5 values
        35, 35, 36, 36, 37
    ]
})


model = TimesFM3Model(
    device="cpu"
)

model.fit(
    targets,
    past_only_covariates=past_only,
    past_future_covariates=past_future,
)

result = model.predict(
    horizon=5,
    return_quantiles=True,
)


print("\nPoint forecast:")
print(result["point_forecast"])

print("\nPoint shape:")
print(
    result[
        "point_forecast"
    ].shape
)

print("\nQuantile shape:")
print(
    result[
        "quantiles"
    ].shape
)