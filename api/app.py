"""
FastAPI application.

Endpoints call the LangGraph workflow or the shared
forecasting service. They do not implement forecasting
math and they do not import the MCP server.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from utils.reproducibility import configure_runtime

from agent.graph import run_forecast_graph
from agent.schemas import ForecastResponse
from api.schemas import (
    DiagnoseRequest,
    ForecastRequest,
    ForecastUrlRequest,
)
from services.data_sources import load_public_csv_url
from services.forecasting_service import (
    diagnose,
    series_from_csv_bytes,
)


configure_runtime(verbose=False)

app = FastAPI(
    title="Agentic Time-Series Forecasting",
    version="1.0.0",
)


def _settings(models):
    settings = {
        "fast": True,
        "n_windows": 1,
        "compute_budget": "low",
    }
    if models:
        settings["model_allowlist"] = list(models)
    return settings


def _response(state):
    forecast = state["forecast_result"]
    selected = forecast["selected_expert"]
    critic = state["critic_result"]
    explanation = critic.get("reason") or forecast.get(
        "evidence",
        "",
    )
    return {
        "selected_model": selected["model"],
        "window_type": selected["window_type"],
        "validation_mae": float(
            selected["validation_mae"]
        ),
        "forecast": [
            float(value)
            for value in forecast["forecast"]
        ],
        "critic_decision": critic["decision"],
        "explanation": explanation,
    }


def _run(values, horizon, target_name, source, models):
    try:
        state = run_forecast_graph(
            user_request=(
                f"Forecast the next {int(horizon)} "
                f"observations of {target_name}. "
                "Use diagnostics, candidate routing, "
                "rolling-origin validation, and dynamic "
                "hard expert selection. Do not invent "
                "forecast numbers."
            ),
            values=list(values),
            horizon=int(horizon),
            dataset_source=source,
            target_column=target_name,
            settings=_settings(models),
        )
    except RuntimeError as exc:
        if "DASHSCOPE" in str(exc):
            raise HTTPException(
                status_code=503,
                detail=str(exc),
            ) from exc
        raise

    return ForecastResponse(
        **_response(state)
    ).model_dump()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/diagnose")
def diagnose_endpoint(body: DiagnoseRequest):
    try:
        return diagnose(
            body.values,
            seasonal_period=body.seasonal_period,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/forecast")
def forecast_endpoint(body: ForecastRequest):
    try:
        return _run(
            values=body.values,
            horizon=body.horizon,
            target_name=body.target_name,
            source="request",
            models=body.models,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/forecast/csv")
async def forecast_csv_endpoint(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    horizon: int = Form(...),
):
    content = await file.read()
    try:
        series = series_from_csv_bytes(
            content,
            file.filename or "",
            target_column,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    try:
        return _run(
            values=series.tolist(),
            horizon=horizon,
            target_name=target_column,
            source="csv-upload",
            models=None,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/forecast/url")
def forecast_url_endpoint(body: ForecastUrlRequest):
    try:
        frame = load_public_csv_url(
            body.url,
            body.target_column,
        )
        values = frame[body.target_column].tolist()
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    try:
        return _run(
            values=values,
            horizon=body.horizon,
            target_name=body.target_column,
            source="url",
            models=body.models,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
