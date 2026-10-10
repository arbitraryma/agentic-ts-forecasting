import numpy as np

from timesfm3 import TimesFM3Evaluator, ModelConfig


# -----------------------------------
# Load TimesFM 3
# -----------------------------------

config = ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    per_core_batch_size=4,
    device="cpu",
)

forecaster = TimesFM3Evaluator(config)


# -----------------------------------
# Univariate context
# -----------------------------------

y = np.array([
    100, 102, 101, 105, 108,
    110, 111, 115, 117, 118,
    120, 122, 125, 127, 129,
    130, 133, 135, 137, 140
], dtype=np.float32)


# -----------------------------------
# Forecast
# -----------------------------------

outputs = list(
    forecaster.predict_batch(
        contexts=[y],
        horizon=5,
        return_quantiles=True,
        use_symmetric_averaging=False,
    )
)

result = outputs[0]


print("\nPoint forecast:")
print(result.forecast)

print("\nPoint forecast shape:")
print(result.forecast.shape)

print("\nQuantile forecast shape:")
print(result.quantiles.shape)