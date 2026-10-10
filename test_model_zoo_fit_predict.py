import argparse
import pandas as pd

from diagnostics.basic import (
    basic_diagnostics,
)

from evaluation.rolling_cv import (
    rolling_backtest,
)

from models.registry import (
    MODEL_REGISTRY,
    get_model_class,
)

from models.config_builder import (
    build_model_kwargs,
)


# ===================================
# Command-line arguments
# ===================================

parser = argparse.ArgumentParser(
    description="Compare forecasting models."
)

parser.add_argument(
    "--fast",
    action="store_true",
    help="Run a lightweight benchmark for debugging.",
)

args = parser.parse_args()


# ===================================
# Load dataset
# ===================================

df = pd.read_csv(
    "data/AirPassengers.csv"
)

y = df[
    "#Passengers"
]


# ===================================
# Diagnostics
# ===================================

diagnostics = basic_diagnostics(
    y
)

print(
    "\n==================================="
)
print(
    "Diagnostics"
)
print(
    "==================================="
)

for key, value in diagnostics.items():
    print(
        f"{key}: {value}"
    )


# ===================================
# Benchmark settings
# ===================================

if args.fast:

    print(
        "\nFAST MODE ENABLED"
    )

    horizon = 2
    n_windows = 1
    train_window = 24

    window_types = [
        "expanding",
    ]

else:

    print(
        "\nFULL MODE ENABLED"
    )

    horizon = 12
    n_windows = 3
    train_window = 60

    window_types = [
        "expanding",
        "sliding",
    ]


print(
    "\nBenchmark settings:"
)

print(
    f"horizon = {horizon}"
)

print(
    f"n_windows = {n_windows}"
)

print(
    f"window_types = {window_types}"
)


# ===================================
# Store results
# ===================================

all_results = []


# ===================================
# Run models
# ===================================

for model_name, config in MODEL_REGISTRY.items():

    # -------------------------------
    # Skip disabled models
    # -------------------------------

    if not config[
        "enabled"
    ]:
        continue


    # -------------------------------
    # Fast mode:
    # skip foundation models
    # -------------------------------

    if (
        args.fast
        and
        config["family"]
        == "foundation"
    ):

        print(
            "\n==================================="
        )

        print(
            f"Skipping {model_name} "
            "in fast mode."
        )

        print(
            "==================================="
        )

        continue


    print(
        "\n==================================="
    )

    print(
        f"Running model: "
        f"{model_name}"
    )

    print(
        "==================================="
    )


    # ===================================
    # Lazy-load model
    # ===================================

    try:

        model_class = (
            get_model_class(
                model_name
            )
        )

    except Exception as e:

        print(
            f"Could not load "
            f"{model_name}: "
            f"{type(e).__name__}: "
            f"{e}"
        )

        continue


    # ===================================
    # Build model kwargs
    # ===================================

    try:

        model_kwargs = (
            build_model_kwargs(
                model_name,
                diagnostics,
            )
        )

    except Exception as e:

        print(
            f"Could not build config "
            f"for {model_name}: "
            f"{type(e).__name__}: "
            f"{e}"
        )

        continue


    # ===================================
    # Fast-mode overrides
    # ===================================

    if args.fast:

        if model_name == "dlinear":

            model_kwargs.update({
                "input_length": 8,
                "horizon": horizon,
                "epochs": 1,
                "batch_size": 32,
            })


        elif model_name == "patchtst":

            model_kwargs.update({
                "input_length": 8,
                "horizon": horizon,
                "epochs": 1,
                "batch_size": 32,
                "patch_length": 4,
                "patch_stride": 4,
            })


        elif model_name == "timesnet":

            model_kwargs.update({
                "input_length": 8,
                "horizon": horizon,
                "max_steps": 1,
            })


        elif model_name == "itransformer":

            model_kwargs.update({
                "input_length": 8,
                "horizon": horizon,
                "n_series": 1,
                "max_steps": 1,
            })


    # ===================================
    # Full-mode overrides
    # ===================================

    else:

        if model_name == "dlinear":

            model_kwargs.update({
                "horizon": horizon,
            })


        elif model_name == "patchtst":

            model_kwargs.update({
                "horizon": horizon,
            })


        elif model_name == "timesnet":

            model_kwargs.update({
                "horizon": horizon,
            })


        elif model_name == "itransformer":

            model_kwargs.update({
                "horizon": horizon,
                "n_series": 1,
            })


    print(
        "Model parameters:"
    )

    print(
        model_kwargs
    )


    # ===================================
    # Run each window type
    # ===================================

    for window_type in window_types:

        print(
            f"\nRunning "
            f"{window_type} CV..."
        )


        try:

            backtest_kwargs = {

                "model_class":
                    model_class,

                "y":
                    y,

                "horizon":
                    horizon,

                "n_windows":
                    n_windows,

                "model_kwargs":
                    model_kwargs,

                "window_type":
                    window_type,
            }


            if (
                window_type
                == "sliding"
            ):

                backtest_kwargs[
                    "train_window"
                ] = train_window


            results = (
                rolling_backtest(
                    **backtest_kwargs
                )
            )


            results[
                "model"
            ] = model_name


            results[
                "family"
            ] = config[
                "family"
            ]


            all_results.append(
                results
            )


            print(
                f"{window_type} CV: OK"
            )


        except Exception as e:

            print(
                f"{window_type} CV "
                f"FAILED for "
                f"{model_name}: "
                f"{type(e).__name__}: "
                f"{e}"
            )


# ===================================
# Check results
# ===================================

if len(
    all_results
) == 0:

    raise RuntimeError(
        "No model completed successfully."
    )


# ===================================
# Combine results
# ===================================

all_results = pd.concat(
    all_results,
    ignore_index=True,
)


print(
    "\n==================================="
)
print(
    "All CV Results"
)
print(
    "==================================="
)

print(
    all_results
)


# ===================================
# Summary
# ===================================

summary = (
    all_results

    .groupby(
        [
            "family",
            "model",
            "window_type",
        ]
    )[
        [
            "mae",
            "rmse",
        ]
    ]

    .mean()

    .reset_index()
)


print(
    "\n==================================="
)
print(
    "Model Comparison"
)
print(
    "==================================="
)

print(
    summary
)


# ===================================
# Ranking
# ===================================

ranking = (
    summary

    .sort_values(
        by="mae"
    )

    .reset_index(
        drop=True
    )
)


print(
    "\n==================================="
)
print(
    "Ranking by MAE"
)
print(
    "==================================="
)

print(
    ranking
)


# ===================================
# Best model
# ===================================

best_row = (
    ranking.iloc[
        0
    ]
)


print(
    "\n==================================="
)
print(
    "Best Model"
)
print(
    "==================================="
)

print(
    f"Family: "
    f"{best_row['family']}"
)

print(
    f"Model: "
    f"{best_row['model']}"
)

print(
    f"Window type: "
    f"{best_row['window_type']}"
)

print(
    f"MAE: "
    f"{best_row['mae']:.4f}"
)

print(
    f"RMSE: "
    f"{best_row['rmse']:.4f}"
)