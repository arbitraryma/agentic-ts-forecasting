import pandas as pd

from models.registry import (
    MODEL_REGISTRY,
    get_model_class,
)

from models.config_builder import (
    build_model_kwargs,
)

from diagnostics.basic import (
    basic_diagnostics,
)


# ===================================
# Small test series
# ===================================

y = pd.Series([
    100, 102, 101, 105, 108,
    110, 111, 115, 117, 118,
    120, 122, 125, 127, 129,
    130, 133, 135, 137, 140,
    142, 145, 147, 149, 151,
    153, 155, 157, 160, 162,
    164, 166, 168, 170, 172,
    175, 177, 180, 182, 185,
])


diagnostics = basic_diagnostics(y)

print("\nDiagnostics:")
print(diagnostics)


# ===================================
# Smoke test
# ===================================

for model_name, config in MODEL_REGISTRY.items():

    if not config["enabled"]:
        continue

    print("\n===================================")
    print(f"Testing: {model_name}")
    print("===================================")

    try:

        model_class = get_model_class(
            model_name
        )

        model_kwargs = build_model_kwargs(
            model_name,
            diagnostics,
        )

        print(
            "Model class:",
            model_class
        )

        print(
            "Model kwargs:",
            model_kwargs
        )

        model = model_class(
            **model_kwargs
        )

        print(
            "Instantiation: OK"
        )

    except Exception as e:

        print(
            f"FAILED: {e}"
        )