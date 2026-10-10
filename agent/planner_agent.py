"""
Planner agent.

LLM reasoning only. The planner decides which
workflow steps are required and returns a JSON
plan. It does not compute forecasts, metrics,
or routing weights.
"""

from langchain_core.tools import tool

from agent.llm import invoke_required_tool
from agent.prompts import PLANNER_TEMPLATE


ALLOWED_STEPS = [
    "diagnostics",
    "candidate_routing",
    "rolling_validation",
    "expert_selection",
    "forecast",
]


@tool
def submit_plan(
    task: str,
    horizon: int,
    steps: list[str],
    reason: str,
) -> str:
    """Submit a forecasting plan. This does not compute a forecast."""

    return "plan_recorded"


def _validate_plan(plan):
    """Reject a plan that is not executable."""

    if not isinstance(plan, dict):
        raise ValueError(
            "Planner output is not a JSON object."
        )

    horizon = int(plan["horizon"])

    if horizon <= 0:
        raise ValueError(
            "Planner horizon must be positive."
        )

    steps = plan["steps"]

    if not isinstance(steps, list) or not steps:
        raise ValueError(
            "Planner steps must be a non-empty list."
        )

    unknown = [
        step
        for step in steps
        if step not in ALLOWED_STEPS
    ]

    if unknown:
        raise ValueError(
            f"Planner returned unknown steps: {unknown}"
        )

    if "forecast" not in steps:
        raise ValueError(
            "A forecast request must include "
            "the forecast step."
        )

    return {
        "task": str(plan["task"]),
        "horizon": horizon,
        "steps": list(steps),
        "reason": str(plan["reason"]),
    }


def planner_agent(user_request):
    """
    LLM planning step.

    Returns the JSON object submitted through
    submit_plan. Numerical forecasting is not
    part of this function.
    """

    def dispatch(tool_name, arguments):

        if tool_name != "submit_plan":
            raise ValueError(
                f"Planner cannot call {tool_name}."
            )

        # The plan is recorded from the tool
        # arguments. Nothing numerical is computed.
        return {
            "status": "plan_recorded",
        }

    _, calls = invoke_required_tool(
        messages=PLANNER_TEMPLATE.format_messages(
            user_request=user_request,
        ),
        tools=[submit_plan],
        dispatch=dispatch,
        stop_tool="submit_plan",
    )

    plans = [
        call["arguments"]
        for call in calls
        if call["name"] == "submit_plan"
    ]

    if not plans:
        raise RuntimeError(
            "Planner did not call submit_plan."
        )

    return _validate_plan(plans[-1])
