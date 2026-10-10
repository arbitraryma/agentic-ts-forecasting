import pandas as pd
import torch

from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module


# simple series
df = pd.DataFrame(
    {
        "target": [
            100, 102, 101, 105, 108,
            110, 111, 115, 117, 118,
            120, 122, 125, 127, 129,
            130, 133, 135, 137, 140,
        ]
    },
    index=pd.date_range(
        start="2020-01-01",
        periods=20,
        freq="D",
    ),
)


dataset = PandasDataset(
    df,
    target="target",
)


module = Moirai2Module.from_pretrained(
    "Salesforce/moirai-2.0-R-small"
)


model = Moirai2Forecast(
    module=module,
    prediction_length=5,
    context_length=20,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0,
)


predictor = model.create_predictor(
    batch_size=1,
)


forecast_it = predictor.predict(
    dataset
)

try:
    forecast = next(forecast_it)
finally:
    forecast_it.close()

median = forecast.quantile(0.5)

print("Forecast median:")
print(median)

print("Shape:")
print(median.shape)