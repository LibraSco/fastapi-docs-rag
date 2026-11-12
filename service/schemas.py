"""HTTP 请求与响应模型。"""
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

SESSION_ID_RE = re.compile(r"[0-9A-Za-z_-]{1,64}")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    session_id: Optional[str] = Field(default=None, max_length=64)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question 不能为空")
        return value

    @field_validator("session_id")
    @classmethod
    def check_session_id(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not SESSION_ID_RE.fullmatch(value):
            raise ValueError("session_id 只允许字母、数字、下划线和短横线，长度 1-64")
        return value


class SourceOut(BaseModel):
    index: int
    chunk_id: str
    source: str
    chunk_index: int
    score: Optional[float] = None
    snippet: str


class MetricsOut(BaseModel):
    llm_calls: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    retrieval_ms: float
    total_ms: float


class ChatResponse(BaseModel):
    request_id: str
    session_id: Optional[str] = None
    answer: str
    refused: bool
    query: str
    citations: list[int]
    sources: list[SourceOut]
    route: str
    grade: str
    verify: str
    tool_choice: str
    trace: list[dict]
    metrics: MetricsOut
