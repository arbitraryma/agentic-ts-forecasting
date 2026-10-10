"""
Qwen client.

LLM reasoning and function calling only.
This module does not compute forecasts, metrics,
or routing weights.

It uses the same environment variables as
agent/router_agent.py:

    DASHSCOPE_API_KEY
    DASHSCOPE_BASE_URL
    QWEN_MODEL
"""

import json

from agent.llm import (
    get_model_name,
    get_openai_client as get_client,
)


def call_qwen(
    instructions,
    user_text,
    tools,
    dispatch,
    stop_tool=None,
):
    """
    LLM function-calling loop.

    `dispatch(tool_name, arguments)` must be a
    deterministic Python callback. This helper
    returns the callback results. It does not
    treat model prose as numerical output.

    The loop matches agent/router_agent.py, with
    one difference: once `stop_tool` has been
    called, the structured tool result is final
    and no extra completion is requested.
    """

    client = get_client()
    model_name = get_model_name()

    response = client.responses.create(
        model=model_name,
        instructions=instructions,
        input=[
            {
                "role": "user",
                "content": user_text,
            }
        ],
        tools=tools,
    )

    captured = []

    while True:

        function_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        if len(function_calls) == 0:
            return (
                response.output_text or "",
                captured,
            )

        tool_outputs = []
        stop = False

        for call in function_calls:

            print(
                f"\n[Agent Tool Call] {call.name}"
            )

            if call.arguments:
                arguments = json.loads(
                    call.arguments
                )
            else:
                arguments = {}

            result = dispatch(
                call.name,
                arguments,
            )

            captured.append({
                "name": call.name,
                "arguments": arguments,
                "result": result,
            })

            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(
                    result,
                    ensure_ascii=False,
                ),
            })

            if (
                stop_tool is not None
                and call.name == stop_tool
            ):
                stop = True

        if stop:
            # The structured tool result is the agent
            # output. Do not ask the model for another
            # completion that could restate numbers.
            return (
                "",
                captured,
            )

        response = client.responses.create(
            model=model_name,
            instructions=instructions,
            previous_response_id=response.id,
            input=tool_outputs,
            tools=tools,
        )
