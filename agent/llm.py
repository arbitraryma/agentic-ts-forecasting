"""
Qwen model access.

LangChain provides the chat model and tool-calling
messages. Numerical forecasting is not done here.

Environment variables:

    DASHSCOPE_API_KEY
    DASHSCOPE_BASE_URL
    QWEN_MODEL
"""

import os

from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from openai import OpenAI


def load_qwen_env():
    """
    Apply non-empty values from .env.

    An empty template must not replace a credential
    already exported in the shell. A filled .env
    does replace a stale shell value.
    """

    for name, value in dotenv_values().items():
        if name not in {
            "DASHSCOPE_API_KEY",
            "DASHSCOPE_BASE_URL",
            "QWEN_MODEL",
        }:
            continue
        text = (value or "").strip()
        if text:
            os.environ[name] = text


load_qwen_env()


def get_model_name():
    return os.getenv(
        "QWEN_MODEL",
        "qwen3.8-flash",
    )


def get_api_key():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set."
        )
    return api_key


def get_base_url():
    base_url = os.getenv("DASHSCOPE_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "DASHSCOPE_BASE_URL is not set."
        )
    return base_url


def get_openai_client():
    """OpenAI-compatible client used by the router agent."""

    return OpenAI(
        api_key=get_api_key(),
        base_url=get_base_url(),
    )


def get_chat_model():
    """LangChain chat model pointed at the Qwen endpoint."""

    return ChatOpenAI(
        model=get_model_name(),
        api_key=get_api_key(),
        base_url=get_base_url(),
        temperature=0,
    )


def invoke_required_tool(
    messages,
    tools,
    dispatch,
    stop_tool,
):
    """
    Ask Qwen to call one tool.

    messages come from a LangChain prompt template.
    The returned payload is the Python dispatch
    result. Model prose is not treated as a forecast.
    """

    model = get_chat_model().bind_tools(tools)
    message = model.invoke(messages)

    captured = []

    for call in message.tool_calls or []:
        name = call["name"]
        arguments = call.get("args") or {}
        result = dispatch(name, arguments)
        captured.append({
            "name": name,
            "arguments": arguments,
            "result": result,
        })
        if name == stop_tool:
            return "", captured

    raise RuntimeError(
        f"Model did not call {stop_tool}."
    )
