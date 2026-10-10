"""
Forecasting agent.

The LLM only selects the forecasting tool.
Diagnostics, candidate routing, rolling
validation, hard expert selection, and the
forecast values are computed by Python tools
in agent/tools.py.
"""

import json

from langchain_core.tools import tool

from agent.llm import invoke_required_tool
from agent.prompts import FORECASTING_TEMPLATE
from agent.tools import (
    run_full_forecasting_pipeline,
)


@tool
def run_pipeline_tool() -> str:
    """Run the deterministic forecasting pipeline in Python.

    The tool computes diagnostics, candidate routing,
    rolling validation, hard expert selection, and the
    forecast. It does not accept forecast numbers.
    """

    return "pipeline_requested"


def forecasting_agent(
    planner_result,
    y,
    retry_feedback=None,
    n_series=1,
    has_covariates=False,
    compute_budget="low",
    n_windows=1,
    window_types=None,
    train_window=60,
    fast=True,
    compute_weight=0.10,
    temperature=1.0,
    known_seasonal_period=None,
    model_allowlist=None,
    verbose=False,
):
    """
    Execute the plan with Python forecasting tools.

    The LLM must call run_full_forecasting_pipeline.
    That tool runs diagnostics, candidate routing,
    rolling validation, hard expert selection, and
    the forecast. The returned JSON is the tool
    result, so the model cannot replace its numbers.
    """

    horizon = int(
        planner_result["horizon"]
    )

    def dispatch(tool_name, arguments):

        if tool_name not in {
            "run_full_forecasting_pipeline",
            "run_pipeline_tool",
        }:
            raise ValueError(
                "Forecasting agent cannot call "
                f"{tool_name}."
            )

        # Arguments from the LLM are ignored.
        # Horizon and data stay under Python control.
        del arguments

        return run_full_forecasting_pipeline(
            y=y,
            horizon=horizon,
            n_series=n_series,
            has_covariates=has_covariates,
            compute_budget=compute_budget,
            n_windows=n_windows,
            window_types=window_types,
            train_window=train_window,
            fast=fast,
            compute_weight=compute_weight,
            temperature=temperature,
            known_seasonal_period=(
                known_seasonal_period
            ),
            model_allowlist=model_allowlist,
            verbose=verbose,
        )

    user_text = json.dumps(
        {
            "plan": planner_result,
            "retry_feedback": retry_feedback,
        },
        ensure_ascii=False,
    )

    _, calls = invoke_required_tool(
        messages=FORECASTING_TEMPLATE.format_messages(
            payload=user_text,
        ),
        tools=[run_pipeline_tool],
        dispatch=dispatch,
        stop_tool="run_pipeline_tool",
    )

    results = [
        call["result"]
        for call in calls
        if call["name"] in {
            "run_pipeline_tool",
            "run_full_forecasting_pipeline",
        }
    ]

    if not results:
        raise RuntimeError(
            "Forecasting agent did not call "
            "run_full_forecasting_pipeline."
        )

    forecast_result = results[-1]

    if len(forecast_result["forecast"]) != horizon:
        raise RuntimeError(
            "Python forecast length does not "
            "match the planned horizon."
        )

    return forecast_result
