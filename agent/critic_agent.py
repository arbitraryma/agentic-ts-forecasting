"""
Critic agent.

Python performs the hard consistency checks.
The LLM then submits ACCEPT or RETRY through
a function call. It does not rewrite the
forecast or the routing scores.

A non-empty deterministic issue list forces
RETRY, so the model cannot accept a broken
result. At most one retry is enforced by
agent/multi_agent.py, not by this file.
"""

import json

from langchain_core.tools import tool

from agent.llm import invoke_required_tool
from agent.prompts import CRITIC_TEMPLATE


@tool
def submit_review(
    decision: str,
    issues: list[str],
    reason: str,
    recommended_action: str = "",
) -> str:
    """Submit ACCEPT or RETRY. This does not change forecast numbers."""

    return "review_recorded"


def check_forecast_result(
    planner_result,
    forecast_result,
):
    """
    Deterministic consistency checks.

    These checks do not create a new forecast.
    """

    issues = []

    horizon = int(
        planner_result["horizon"]
    )

    forecast = forecast_result.get(
        "forecast"
    )

    if not isinstance(forecast, list):
        issues.append(
            "Forecast output is missing."
        )
    elif len(forecast) != horizon:
        issues.append(
            "Forecast length "
            f"{len(forecast)} does not match "
            f"horizon {horizon}."
        )

    validation = forecast_result.get(
        "validation_summary"
    ) or []

    if len(validation) == 0:
        issues.append(
            "Validation results are missing."
        )

    selected = forecast_result.get(
        "selected_expert"
    ) or {}

    model = selected.get("model")
    window_type = selected.get(
        "window_type"
    )

    if selected.get("validation_mae") is None:
        issues.append(
            "Selected expert has no "
            "validation MAE."
        )

    if not forecast_result.get("evidence"):
        issues.append(
            "Selection evidence is missing."
        )

    candidates = forecast_result.get(
        "candidates"
    ) or []

    if model not in candidates:
        issues.append(
            "Selected model is not in the "
            "candidate set."
        )

    matched = [
        row
        for row in validation
        if row.get("model") == model
        and row.get("window_type") == window_type
    ]

    if not matched:
        issues.append(
            "Selected expert is not supported "
            "by a validation row."
        )
    else:
        row_mae = matched[0].get("mae")
        selected_mae = selected.get(
            "validation_mae"
        )

        if (
            row_mae is not None
            and selected_mae is not None
            and abs(
                float(row_mae)
                - float(selected_mae)
            ) > 1e-4
        ):
            issues.append(
                "Selected validation MAE does "
                "not match the validation table."
            )

    return issues


def critic_agent(
    planner_result,
    forecast_result,
):
    """
    Review one forecasting result.

    Returns ACCEPT, or RETRY with a reason
    and a recommended action.
    """

    issues = check_forecast_result(
        planner_result,
        forecast_result,
    )

    def dispatch(tool_name, arguments):

        if tool_name != "submit_review":
            raise ValueError(
                f"Critic cannot call {tool_name}."
            )

        return {
            "status": "review_recorded",
        }

    user_text = json.dumps(
        {
            "plan": planner_result,
            "forecast_result": forecast_result,
            "deterministic_issues": issues,
        },
        ensure_ascii=False,
    )

    _, calls = invoke_required_tool(
        messages=CRITIC_TEMPLATE.format_messages(
            payload=user_text,
        ),
        tools=[submit_review],
        dispatch=dispatch,
        stop_tool="submit_review",
    )

    reviews = [
        call["arguments"]
        for call in calls
        if call["name"] == "submit_review"
    ]

    if not reviews:
        raise RuntimeError(
            "Critic did not call submit_review."
        )

    review = reviews[-1]
    decision = str(
        review.get("decision", "")
    ).upper()

    if decision not in {"ACCEPT", "RETRY"}:
        raise ValueError(
            f"Invalid critic decision: {decision}"
        )

    model_issues = review.get("issues") or []

    if not isinstance(model_issues, list):
        model_issues = [str(model_issues)]

    # Deterministic failures override an ACCEPT.
    # The LLM cannot approve a broken forecast.
    if issues and decision == "ACCEPT":
        decision = "RETRY"

    combined_issues = list(issues)

    for item in model_issues:
        text = str(item)
        if text not in combined_issues:
            combined_issues.append(text)

    if decision == "RETRY":

        action = review.get(
            "recommended_action"
        )

        if not action:
            action = (
                "Rerun run_full_forecasting_pipeline "
                "so Python recomputes validation, "
                "expert selection, and the forecast."
            )

        return {
            "decision": "RETRY",
            "issues": combined_issues,
            "reason": str(
                review.get("reason", "")
            ),
            "recommended_action": str(action),
        }

    return {
        "decision": "ACCEPT",
        "issues": [],
        "reason": str(
            review.get("reason", "")
        ),
    }
