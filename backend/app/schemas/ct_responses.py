# backend/app/schemas/ct_responses.py

from typing import List
from pydantic import BaseModel


class CTPredictionResponse(BaseModel):
    predicted_class: int
    probabilities: List[float]


class CTExplainResponse(BaseModel):
    predicted_class: int
    probabilities: List[float]
    heatmap_base64: str
