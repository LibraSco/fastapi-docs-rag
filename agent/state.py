"""图状态与轨迹结构。

所有节点只返回自己改动的那部分 key；计数器和轨迹用 operator.add 累加，
避免节点之间互相覆盖。初始状态必须带 0 值，否则累加会从缺失开始。
"""
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, TypedDict

from vector_stores import Hit


@dataclass
class TraceStep:
    node: str
    ms: float
    detail: str = ""

    def as_dict(self) -> dict:
        return {"node": self.node, "ms": round(self.ms, 1), "detail": self.detail}


@dataclass
class AgentAnswer:
    text: str
    sources: list[Hit] = field(default_factory=list)
    first_sources: list[Hit] = field(default_factory=list)
    query: str = ""
    refused: bool = False
    citations: list[int] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_ms: float = 0.0
    route: str = ""
    grade: str = ""
    verify: str = ""
    tool_choice: str = ""

    def to_dict(self) -> dict:
        def source_payload(index: int, hit: Hit) -> dict:
            score = hit.rerank_score
            if score is None and hit.distance > 0:
                score = 1.0 / hit.distance - 1.0
            return {
                "index": index,
                "chunk_id": hit.chunk_id,
                "source": hit.source,
                "chunk_index": hit.chunk_index,
                "score": round(float(score), 3) if score is not None else None,
                "snippet": hit.text[:300],
            }

        retrieval_ms = sum(
            step.get("ms", 0) for step in self.trace if step.get("node") in ("select_tool", "retrieve", "rerank")
        )
        return {
            "answer": self.text,
            "refused": self.refused,
            "query": self.query,
            "citations": self.citations,
            "sources": [source_payload(i, hit) for i, hit in enumerate(self.sources, 1)],
            "route": self.route,
            "grade": self.grade,
            "verify": self.verify,
            "tool_choice": self.tool_choice,
            "trace": self.trace,
            "metrics": {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "retrieval_ms": round(retrieval_ms, 1),
                "total_ms": round(self.total_ms, 1),
            },
        }


class AgentState(TypedDict, total=False):
    question: str
    history: Any                    # list[BaseMessage]，会话历史由外部传入
    standalone: str                 # 结合历史补全后的问题，精排和判定始终围绕它
    search_query: str               # 本轮召回用的查询，重查时会被改写
    route: str                      # answer / single / multi
    sub_questions: list[str]        # 多跳拆解出的子问题
    queries: list[str]              # 实际执行过的检索 query
    hits: list[Hit]
    candidates: list[Hit]           # 历轮召回并集，重查时只增不减
    first_hits: list[Hit]           # 第一轮精排结果，用于评测重查是否救回/误伤
    grade: str                      # sufficient / insufficient
    attempt: int                    # 已执行的检索轮次
    answer: str
    refused: bool
    citations: list[int]
    verify: str                     # pass / unsupported / skipped
    verify_attempt: int             # 已执行的校验轮次，防止重生成死循环
    verify_reason: str              # 校验不通过时列出的无依据结论，用于重生成
    tool_choice: str                # vector / bm25 / both
    trace: Annotated[list[dict], operator.add]
    llm_calls: Annotated[int, operator.add]
    tool_calls: Annotated[int, operator.add]
    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]
    error: str


def initial_state(question: str, history: Optional[list] = None, route: str = "single") -> dict:
    return {
        "question": question,
        "history": history or [],
        "standalone": question,
        "search_query": question,
        "route": route,
        "sub_questions": [],
        "queries": [],
        "hits": [],
        "candidates": [],
        "first_hits": [],
        "grade": "",
        "attempt": 0,
        "answer": "",
        "refused": False,
        "citations": [],
        "verify": "skipped",
        "verify_attempt": 0,
        "verify_reason": "",
        "tool_choice": "vector",
        "trace": [],
        "llm_calls": 0,
        "tool_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": "",
    }
