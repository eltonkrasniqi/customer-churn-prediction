"""Pydantic schemas for API request/response validation."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class ChannelEnum(str, Enum):
    """Valid channel types."""
    PHONE = "phone"
    EMAIL = "email"
    CHAT = "chat"


class PlanTierEnum(str, Enum):
    """Valid plan tiers."""
    BASIC = "basic"
    PLUS = "plus"
    PRO = "pro"


class ChurnPredictionRequest(BaseModel):
    """Request schema for churn prediction."""
    
    tenure_days: int = Field(
        ...,
        ge=1,
        le=10000,
        description="Number of days customer has been active"
    )
    tickets_last_30d: int = Field(
        ...,
        ge=0,
        le=100,
        description="Number of support tickets in last 30 days"
    )
    avg_handle_time: float = Field(
        ...,
        ge=0,
        le=3600,
        description="Average handle time in seconds"
    )
    first_contact_resolution: Literal[0, 1] = Field(
        ...,
        description="Whether issue was resolved on first contact (0 or 1)"
    )
    sentiment_avg: float = Field(
        ...,
        ge=-3.0,
        le=3.0,
        description="Average sentiment score (-3 to +3)"
    )
    escalations_90d: int = Field(
        ...,
        ge=0,
        le=50,
        description="Number of escalations in last 90 days"
    )
    channel: ChannelEnum = Field(
        ...,
        description="Primary support channel"
    )
    plan_tier: PlanTierEnum = Field(
        ...,
        description="Customer plan tier"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenure_days": 120,
                "tickets_last_30d": 5,
                "avg_handle_time": 650.0,
                "first_contact_resolution": 0,
                "sentiment_avg": -1.2,
                "escalations_90d": 2,
                "channel": "email",
                "plan_tier": "basic"
            }
        }
    )


class ChurnPredictionResponse(BaseModel):
    """Response schema for churn prediction."""
    
    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability of churn (0-1)"
    )
    risk_band: Literal["Low", "Medium", "High"] = Field(
        ...,
        description="Risk classification band"
    )
    model_version: str = Field(
        ...,
        description="Model version identifier"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "churn_probability": 0.7312,
                "risk_band": "High",
                "model_version": "20240101_1200"
            }
        }
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok"
            }
        }
    )


class ReadyResponse(BaseModel):
    ready: bool = Field(..., description="Whether model is loaded")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ready": True
            }
        }
    )
