import json

from analysis.router_summary import (
    summarize_router_sensitivity,
)


summary = summarize_router_sensitivity(
    results_dir="results"
)


print(
    json.dumps(
        summary,
        indent=2,
    )
)
