import json
import os

from openai import OpenAI

from agent.tools import (
    execute_tool,
)


# =========================================================
# Environment configuration
# =========================================================

API_KEY = os.getenv(
    "DASHSCOPE_API_KEY"
)

BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL"
)


if not API_KEY:
    raise RuntimeError(
        "DASHSCOPE_API_KEY is not set."
    )


if not BASE_URL:
    raise RuntimeError(
        "DASHSCOPE_BASE_URL is not set."
    )


# =========================================================
# Qwen client
# =========================================================

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


# =========================================================
# Model
#
# You can change this later depending on cost.
# Start with a Flash model for this GitHub demo.
# =========================================================

MODEL_NAME = os.getenv(
    "QWEN_MODEL",
    "qwen3.8-flash",
)


# =========================================================
# Function tools
# =========================================================

TOOLS = [

    {
        "type": "function",

        "name":
            "run_router_sensitivity",

        "description":
            (
                "Run the deterministic router sensitivity "
                "experiment. Use this only when fresh "
                "sensitivity results are needed or the "
                "existing result files are unavailable."
            ),

        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },

    {
        "type": "function",

        "name":
            "summarize_router_results",

        "description":
            (
                "Read existing router sensitivity results "
                "and return a compact structured summary. "
                "The summary contains winner stability, "
                "routing entropy, maximum routing weight, "
                "effective sample size, and sensitivity "
                "to bandwidth, temperature, recency decay, "
                "and number of rolling windows."
            ),

        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]


# =========================================================
# Prompt engineering
# =========================================================

SYSTEM_PROMPT = """
You are the router-analysis agent in an adaptive
time-series forecasting system.

Your role is to analyze deterministic sensitivity
experiments and recommend defensible routing
hyperparameters.

The numerical calculations are performed by Python
tools. You must not invent numerical values.

When analyzing the router, consider:

1. Winner stability across forecast horizons.
2. Routing entropy.
3. Maximum routing weight.
4. Effective sample size (ESS).
5. Sensitivity to state bandwidth.
6. Sensitivity to temperature.
7. Sensitivity to recency decay.
8. Sensitivity to the number of rolling windows.

Interpretation guidelines:

- Very low ESS suggests that routing relies on too few
  historical validation states.

- Very high maximum routing weight may indicate overly
  sharp or unstable routing.

- Higher routing entropy generally indicates a more
  diversified ensemble, but high entropy is not
  automatically better.

- A good configuration should balance predictive
  discrimination with robustness and adequate ESS.

- The descriptive stability score is only a diagnostic
  aid. Do not treat it as a mathematically proven optimum.

- Clearly distinguish empirical evidence from your own
  interpretation.

- If the existing evidence is insufficient, explicitly
  state the limitation.

Tool-use policy:

- First try to summarize existing router results.
- Do not rerun the expensive sensitivity experiment
  unless the summary tool fails because results are
  missing or the user explicitly requests a fresh run.

Your final answer should provide:

1. Recommended state bandwidth.
2. Recommended temperature.
3. Recommended recency decay.
4. Recommended number of rolling windows.
5. Evidence supporting each recommendation.
6. Important robustness concerns.
7. A concise final configuration suitable for the
   forecasting pipeline.
"""


# =========================================================
# Agent
# =========================================================

def analyze_router(
    user_request,
):
    """
    Run the Qwen router-analysis agent.

    The model can call deterministic Python tools and
    reason over the returned structured results.
    """

    response = client.responses.create(

        model=MODEL_NAME,

        instructions=SYSTEM_PROMPT,

        input=[
            {
                "role": "user",
                "content": user_request,
            }
        ],

        tools=TOOLS,
    )


    # =====================================================
    # Function-calling loop
    # =====================================================

    while True:

        function_calls = [

            item

            for item in response.output

            if item.type
            == "function_call"
        ]


        # -------------------------------------------------
        # No more function calls:
        # Qwen has produced the final answer.
        # -------------------------------------------------

        if len(
            function_calls
        ) == 0:

            return response.output_text


        tool_outputs = []


        # -------------------------------------------------
        # Execute requested functions
        # -------------------------------------------------

        for call in function_calls:

            print(
                f"\n[Agent Tool Call] "
                f"{call.name}"
            )


            if call.arguments:

                arguments = json.loads(
                    call.arguments
                )

            else:

                arguments = {}


            result = execute_tool(

                tool_name=
                    call.name,

                arguments=
                    arguments,
            )


            tool_outputs.append({

                "type":
                    "function_call_output",

                "call_id":
                    call.call_id,

                "output":
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    ),
            })


        # -------------------------------------------------
        # Send tool results back to Qwen
        # -------------------------------------------------

        response = client.responses.create(

            model=MODEL_NAME,

            instructions=SYSTEM_PROMPT,

            previous_response_id=
                response.id,

            input=
                tool_outputs,

            tools=
                TOOLS,
        )


# =========================================================
# CLI demo
# =========================================================

if __name__ == "__main__":

    request = """
Analyze the existing router sensitivity experiment.

Use the existing result files if they are available.
Do not rerun the sensitivity experiment unless needed.

Recommend suitable values for:

- state bandwidth
- temperature
- recency decay
- number of rolling windows

Base the recommendation on:

- winner stability
- routing entropy
- maximum routing weight
- effective sample size

Explain the trade-offs and identify any robustness
concerns.

Finish with one recommended configuration that can be
used as the default router setting.
"""


    result = analyze_router(
        request
    )


    print(
        "\n==================================="
    )

    print(
        "Qwen Router Analysis"
    )

    print(
        "==================================="
    )

    print(
        result
    )