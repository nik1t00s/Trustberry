from pydantic import BaseModel, Field
from typing import Optional


class ReviewIn(BaseModel):
    product_id: Optional[int] = Field(default=None)
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    color: Optional[str] = None
    review_text: str
    seller_answer: Optional[str] = None


class PredictionOut(BaseModel):
    prediction: int
    label: str
    probability_fake: float
    model_version: str | None = None
    model_type: str | None = None


class TrainResponse(BaseModel):
    message: str
    train_size: int
    test_size: int
    metrics: dict
