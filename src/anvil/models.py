from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Tool(BaseModel):
    name: str
    description: str
    input_schema: dict
    implementation: str
    trigger_embedding: list[float] = Field(default_factory=list)
    created_from_query: str = ""
    usage_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)


class QueryResult(BaseModel):
    query: str
    output: str
    path_taken: Literal["local_direct", "cache_hit", "generated", "fallback"]
    matched_tool_name: Optional[str] = None
    similarity_score: Optional[float] = None
    latency_ms: float
    trace: list[str] = Field(default_factory=list)
