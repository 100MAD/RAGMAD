import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ChatCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ChatUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ChatOut(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentOut(BaseModel):
    id: uuid.UUID
    chat_id: uuid.UUID
    filename: str
    content_type: str
    size: int
    status: str
    error: str | None
    chunk_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class SourceOut(BaseModel):
    document_id: str | None = None
    filename: str | None = None
    page: int | None = None
    text: str
    score: float | None = None


class MessageCreate(BaseModel):
    content: str = Field(min_length=1)


class MessageOut(BaseModel):
    id: uuid.UUID
    chat_id: uuid.UUID
    role: str
    content: str
    sources: list[SourceOut] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class MetricStatsOut(BaseModel):
    label: str
    mean: float
    min: float
    max: float


class EvaluationSummaryOut(BaseModel):
    question_count: int
    scored_count: int
    overall_score: float | None = None
    metrics: dict[str, MetricStatsOut]


class EvaluationQuestionOut(BaseModel):
    question: str
    response: str
    reference: str
    retrieved_contexts: str | list[str] | None = None
    scores: dict[str, float] = Field(default_factory=dict)


class EvaluationRunSummaryOut(BaseModel):
    id: str
    created_at: datetime
    config: dict
    summary: EvaluationSummaryOut


class EvaluationRunOut(EvaluationRunSummaryOut):
    questions: list[EvaluationQuestionOut]
