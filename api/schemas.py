"""Request models for the REST API."""

from pydantic import BaseModel, Field


class DiagnoseRequest(BaseModel):
    values: list[float]
    seasonal_period: int | None = None


class ForecastRequest(BaseModel):
    values: list[float]
    horizon: int = Field(gt=0)
    target_name: str = "series"
    models: list[str] | None = None


class ForecastUrlRequest(BaseModel):
    url: str
    target_column: str
    horizon: int = Field(gt=0)
    models: list[str] | None = None
