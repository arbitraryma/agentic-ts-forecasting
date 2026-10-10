import pandas as pd

from models.foundation.moirai2 import (
    Moirai2Model,
)


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
    ],

})


model = Moirai2Model(
    freq="D"
)

model.fit(
    targets
)

result = model.predict(
    horizon=5,
    return_quantiles=True,
)


print(
    "\nPoint forecast:"
)
print(
    result["point_forecast"]
)


print(
    "\nPoint shape:"
)
print(
    result[
        "point_forecast"
    ].shape
)


print(
    "\nQuantile shape:"
)
print(
    result[
        "quantiles"
    ].shape
)


print(
    "\nQuantile levels:"
)
print(
    result[
        "quantile_levels"
    ]
)