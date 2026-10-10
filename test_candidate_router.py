import pandas as pd

from diagnostics.basic import basic_diagnostics

from routing.candidate_router import (
    route_candidates,
)


# ===================================
# Example series
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
    187, 190, 192, 194, 196,
    198, 200, 202, 205, 207,
])


diagnostics = basic_diagnostics(
    y
)


candidates, reasons = route_candidates(
    diagnostics=diagnostics,
    horizon=12,
    n_series=1,
    has_covariates=False,
    compute_budget="medium",
)


print(
    "\nDiagnostics:"
)

print(
    diagnostics
)


print(
    "\nSelected candidates:"
)

for model_name in candidates:

    print(
        f"\n{model_name}"
    )

    for reason in reasons[
        model_name
    ]:

        print(
            f"  - {reason}"
        )