"""
LangGraph workflow.

START -> planner -> forecasting -> critic
                              -> ACCEPT -> END
                              -> RETRY  -> forecasting once

The nodes call the existing agents. Those agents
call deterministic tools. This file only controls
the order and the single retry.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agent.critic_agent import critic_agent
from agent.forecasting_agent import forecasting_agent
from agent.planner_agent import planner_agent


class ForecastState(TypedDict, total=False):
    user_request: str
    dataset_source: str
    target_column: str
    horizon: int
    values: list
    plan: dict
    forecast_result: dict
    critic_result: dict
    retry_count: int
    settings: dict


def planner_node(state: ForecastState):
    """LLM planning. A caller-supplied horizon stays fixed."""

    plan = planner_agent(state["user_request"])

    if state.get("horizon"):
        plan = dict(plan)
        plan["horizon"] = int(state["horizon"])

    return {
        "plan": plan,
        "horizon": int(plan["horizon"]),
    }


def forecasting_node(state: ForecastState):
    """Run the deterministic pipeline through the forecasting agent."""

    retry_count = int(state.get("retry_count") or 0)
    feedback = None
    critic_result = state.get("critic_result") or {}

    if critic_result.get("decision") == "RETRY":
        feedback = critic_result
        retry_count += 1

    forecast_result = forecasting_agent(
        state["plan"],
        state["values"],
        retry_feedback=feedback,
        **(state.get("settings") or {}),
    )

    return {
        "forecast_result": forecast_result,
        "retry_count": retry_count,
    }


def critic_node(state: ForecastState):
    """Review the tool result. Does not rewrite the forecast."""

    return {
        "critic_result": critic_agent(
            state["plan"],
            state["forecast_result"],
        )
    }


def route_after_critic(state: ForecastState):
    """
    Allow one retry.

    retry_count is incremented when forecasting runs
    again, so a second RETRY cannot re-enter the node.
    """

    decision = (
        state.get("critic_result") or {}
    ).get("decision")

    if (
        decision == "RETRY"
        and int(state.get("retry_count") or 0) < 1
    ):
        return "forecasting"

    return END


def build_graph():
    graph = StateGraph(ForecastState)
    graph.add_node("planner", planner_node)
    graph.add_node("forecasting", forecasting_node)
    graph.add_node("critic", critic_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "forecasting")
    graph.add_edge("forecasting", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "forecasting": "forecasting",
            END: END,
        },
    )
    return graph.compile()


WORKFLOW = build_graph()


def run_forecast_graph(
    user_request,
    values,
    horizon=None,
    dataset_source="series",
    target_column=None,
    settings=None,
):
    """Execute the compiled graph once."""

    state = {
        "user_request": user_request,
        "values": [float(value) for value in values],
        "dataset_source": dataset_source,
        "target_column": target_column,
        "retry_count": 0,
        "settings": dict(settings or {}),
    }

    if horizon is not None:
        state["horizon"] = int(horizon)

    return WORKFLOW.invoke(state)
