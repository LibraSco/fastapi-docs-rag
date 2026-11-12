"""检索工具：把召回与精排包装成编排层可调用的最小单元。

编排层只认 ToolResult，不关心底层是向量库还是关键词索引；
后续加 BM25 时新增一个同样签名的工具即可。
"""
import logging
import re
import time
from dataclasses import dataclass, field

import jieba
from rank_bm25 import BM25Okapi

import config_data as config
from rerank import RerankService
from vector_stores import Hit, VectorStoreService

jieba.setLogLevel(logging.WARNING)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9_.\-]+")


def tokenize(text: str) -> list[str]:
    """中文用 jieba，英文/标识符整体保留；查询和文档共用同一套切词。"""
    tokens = []
    for part in re.split(r"([A-Za-z0-9_.\-]+)", text or ""):
        if not part:
            continue
        if IDENTIFIER_RE.fullmatch(part):
            tokens.append(part.lower())
            continue
        tokens.extend(
            token.lower()
            for token in jieba.lcut(part)
            if CJK_RE.search(token) or token.isalnum()
        )
    return tokens


@dataclass
class ToolResult:
    name: str
    query: str
    hits: list[Hit] = field(default_factory=list)
    ms: float = 0.0


def dedupe_hits(groups: list[list[Hit]]) -> list[Hit]:
    """按 chunk_id 合并多路子查询结果，同一片段保留向量距离更近的那条。"""
    merged: dict[str, Hit] = {}
    for hits in groups:
        for hit in hits:
            current = merged.get(hit.chunk_id)
            if current is None or hit.distance < current.distance:
                merged[hit.chunk_id] = hit
    return list(merged.values())


def source_anchor_hits(groups: list[list[Hit]], chosen: list[Hit]) -> list[Hit]:
    """多跳兜底：每个子查询若有一篇来源没进最终上下文，补它召回最好的那条。

    精排按整句打分时，可能把某个子问题的文档整体压低；这里只补来源缺口，
    不重排已选片段，避免把单跳问题的排序也改掉。
    """
    covered = {hit.source for hit in chosen}
    anchors = []
    for hits in groups:
        for hit in hits:
            if hit.source not in covered:
                anchors.append(hit)
                covered.add(hit.source)
                break
    return anchors


class VectorTool:
    name = "vector_search"
    description = "语义检索：适合概念、用法、为什么这类自然语言问题"

    def __init__(self, vector_service=None, reranker=None):
        self.vector_service = vector_service or VectorStoreService()
        self.reranker = reranker or RerankService()

    def recall(self, query: str, k: int = None) -> ToolResult:
        """只召回不精排；多路子查询先合并，再统一精排，避免各自截断丢互补片段。"""
        start = time.perf_counter()
        hits = self.vector_service.search(query, k=k or config.recall_k)
        return ToolResult(self.name, query, hits, (time.perf_counter() - start) * 1000)

    def search(self, query: str, recall_k: int = None, final_k: int = None) -> ToolResult:
        start = time.perf_counter()
        hits = self.vector_service.search(query, k=recall_k or config.recall_k)
        hits = self.reranker.rerank(query, hits, top_n=final_k or config.final_k)
        return ToolResult(self.name, query, hits, (time.perf_counter() - start) * 1000)

    def rerank(self, query: str, hits: list[Hit], top_n: int = None) -> ToolResult:
        start = time.perf_counter()
        ranked = self.reranker.rerank(query, hits, top_n=top_n or config.final_k)
        return ToolResult(self.name, query, ranked, (time.perf_counter() - start) * 1000)


class BM25Tool:
    name = "bm25_search"
    description = "关键词检索：适合精确 API 名、参数名、版本号、报错信息"

    def __init__(self, vector_service=None):
        self.vector_service = vector_service or VectorStoreService()
        data = self.vector_service.vector_store.get(include=["documents", "metadatas"])
        self.chunks = []
        for text, meta in zip(data["documents"], data["metadatas"]):
            meta = meta or {}
            self.chunks.append(
                Hit(
                    chunk_id=meta.get("chunk_id", ""),
                    text=text,
                    source=meta.get("source", "unknown"),
                    chunk_index=int(meta.get("chunk_index", -1)),
                )
            )
        # 索引直接建在 Chroma 的同一批 chunk 上，chunk_id 与向量检索完全一致
        self.bm25 = BM25Okapi([tokenize(chunk.text) for chunk in self.chunks])

    def top_score(self, query: str) -> float:
        tokens = tokenize(query)
        if not tokens:
            return 0.0
        return float(max(self.bm25.get_scores(tokens)))

    def search(self, query: str, k: int = None) -> ToolResult:
        start = time.perf_counter()
        tokens = tokenize(query)
        if not tokens:
            return ToolResult(self.name, query, [], (time.perf_counter() - start) * 1000)
        scores = self.bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[: k or config.recall_k]
        hits = [
            Hit(
                chunk_id=self.chunks[index].chunk_id,
                text=self.chunks[index].text,
                source=self.chunks[index].source,
                chunk_index=self.chunks[index].chunk_index,
                distance=1.0 / (1.0 + float(scores[index])),
            )
            for index in order
            if scores[index] > 0
        ]
        return ToolResult(self.name, query, hits, (time.perf_counter() - start) * 1000)
