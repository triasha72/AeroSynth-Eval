"""Confidence calibration metrics and simple post-hoc temperature fitting."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field


class CalibrationPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    confidence: float = Field(ge=0.0, le=1.0)
    correct: bool


class CalibrationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    count: int = Field(ge=1)
    expected_calibration_error: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0, le=1.0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    empirical_accuracy: float = Field(ge=0.0, le=1.0)


def summarize_calibration(points: list[CalibrationPoint], *, bins: int = 10) -> CalibrationSummary:
    if not points:
        raise ValueError("At least one calibration point is required.")
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [
            p
            for p in points
            if lower <= p.confidence < upper or (index == bins - 1 and p.confidence == 1.0)
        ]
        if not bucket:
            continue
        bucket_confidence = sum(p.confidence for p in bucket) / len(bucket)
        bucket_accuracy = sum(p.correct for p in bucket) / len(bucket)
        ece += len(bucket) / len(points) * abs(bucket_accuracy - bucket_confidence)
    brier = sum((p.confidence - float(p.correct)) ** 2 for p in points) / len(points)
    return CalibrationSummary(
        count=len(points),
        expected_calibration_error=ece,
        brier_score=brier,
        mean_confidence=sum(p.confidence for p in points) / len(points),
        empirical_accuracy=sum(p.correct for p in points) / len(points),
    )


def apply_temperature(confidence: float, temperature: float) -> float:
    if not 0.0 < confidence < 1.0:
        return confidence
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    logit = math.log(confidence / (1.0 - confidence))
    return 1.0 / (1.0 + math.exp(-logit / temperature))


def fit_temperature(points: list[CalibrationPoint]) -> float:
    if not points:
        raise ValueError("At least one calibration point is required.")
    candidates = [0.25 + 0.05 * index for index in range(76)]

    def loss(temp: float) -> float:
        total = 0.0
        for point in points:
            probability = min(max(apply_temperature(point.confidence, temp), 1e-6), 1 - 1e-6)
            target = float(point.correct)
            total += -(target * math.log(probability) + (1 - target) * math.log(1 - probability))
        return total / len(points)

    return min(candidates, key=loss)
