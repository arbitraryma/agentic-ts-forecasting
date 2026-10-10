import pandas as pd

from models.foundation.chronos2 import Chronos2Model


# Simple univariate time series
y = pd.Series([
    100, 102, 101, 105, 108,
    110, 111, 115, 117, 118,
    120, 122, 125, 127, 129,
    130, 133, 135, 137, 140
])


model = Chronos2Model(
    device="cpu"
)

model.fit(y)

forecast = model.predict(
    horizon=5
)

print("\nForecast:")
print(forecast)

print("\nType:")
print(type(forecast))

print("\nShape:")
print(forecast.shape)