import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

from models.seasonal_naive import SeasonalNaive
from evaluation.rolling_cv import rolling_backtest


# =========================================================
# 1. Load data
# =========================================================

df = pd.read_csv("data/AirPassengers.csv")

df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month")

y = df["#Passengers"]


print("First rows:")
print(df.head())

print("\nData shape:")
print(df.shape)

print("\nMissing values:")
print(df.isna().sum())


# =========================================================
# 2. Plot raw time series
# =========================================================

plt.figure(figsize=(12, 5))

plt.plot(
    df["Month"],
    df["#Passengers"]
)

plt.xlabel("Time")
plt.ylabel("Passengers")
plt.title("AirPassengers Time Series")

plt.tight_layout()
plt.show()


# =========================================================
# 3. ACF
# =========================================================

plot_acf(
    y,
    lags=36
)

plt.title("Autocorrelation of AirPassengers")

plt.tight_layout()
plt.show()


# =========================================================
# 4. Seasonal decomposition
# =========================================================

series = df.set_index("Month")["#Passengers"]

decomposition = seasonal_decompose(
    series,
    model="multiplicative",
    period=12
)

decomposition.plot()

plt.tight_layout()
plt.show()


# =========================================================
# 5. Forecasting configuration
# =========================================================

horizon = 12
n_windows = 3
season_length = 12

# Sliding-window training length
# 60 months = 5 years
train_window = 60


# =========================================================
# 6. Expanding-window rolling CV
# =========================================================

results_expanding = rolling_backtest(
    model_class=SeasonalNaive,
    y=y,
    horizon=horizon,
    n_windows=n_windows,
    model_kwargs={
        "season_length": season_length
    },
    window_type="expanding"
)


# =========================================================
# 7. Sliding-window rolling CV
# =========================================================

results_sliding = rolling_backtest(
    model_class=SeasonalNaive,
    y=y,
    horizon=horizon,
    n_windows=n_windows,
    model_kwargs={
        "season_length": season_length
    },
    window_type="sliding",
    train_window=train_window
)


# =========================================================
# 8. Print CV results
# =========================================================

print("\n===================================")
print("Expanding-window CV")
print("===================================")

print(results_expanding)


print("\n===================================")
print("Sliding-window CV")
print("===================================")

print(results_sliding)


# =========================================================
# 9. Average metrics
# =========================================================

expanding_avg = (
    results_expanding[
        ["mae", "rmse"]
    ]
    .mean()
)

sliding_avg = (
    results_sliding[
        ["mae", "rmse"]
    ]
    .mean()
)


print("\n===================================")
print("Average metrics")
print("===================================")

print("\nExpanding window:")
print(expanding_avg)

print("\nSliding window:")
print(sliding_avg)


# =========================================================
# 10. Simple comparison
# =========================================================

comparison = pd.DataFrame({
    "expanding": expanding_avg,
    "sliding": sliding_avg
})

print("\n===================================")
print("Comparison")
print("===================================")

print(comparison)