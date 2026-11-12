"""MCP server：把知识库的两个检索器暴露成标准 MCP tools（stdio）。

用法：
    python mcp_server.py

在支持 MCP 的客户端里配置 command 为该 Python 解释器、args 为本文件路径即可。
工具只做检索，不调用生成模型；问答编排仍由 FastAPI 服务负责。
"""
from mcp.server.fastmcp import FastMCP

from agent.tools import BM25Tool, VectorTool

mcp = FastMCP("fastapi-docs-rag")

_vector = None
_bm25 = None


def _tools() -> tuple[VectorTool, BM25Tool]:
    global _vector, _bm25
    if _vector is None:
        _vector = VectorTool()
        _bm25 = BM25Tool(vector_service=_vector.vector_service)
    return _vector, _bm25


def _clamp(k: int) -> int:
    return max(1, min(int(k), 10))


def _hit_payload(hit) -> dict:
    score = hit.rerank_score
    if score is None:
        score = 1.0 / hit.distance - 1.0 if hit.distance > 0 else 0.0
    return {
        "source": hit.source,
        "chunk_index": hit.chunk_index,
        "chunk_id": hit.chunk_id,
        "score": round(float(score), 3),
        "text": hit.text[:800],
    }


@mcp.tool()
def search_docs(query: str, k: int = 5) -> list[dict]:
    """语义检索 FastAPI 中文文档。适合“怎么做”“为什么”这类自然语言问题。

    Args:
        query: 检索问句，中文即可。
        k: 返回片段数，1-10。
    """
    if not query.strip():
        return []
    vector, _ = _tools()
    result = vector.search(query, final_k=_clamp(k))
    return [_hit_payload(hit) for hit in result.hits]


@mcp.tool()
def bm25_search(query: str, k: int = 5) -> list[dict]:
    """关键词检索 FastAPI 中文文档。适合精确 API 名、参数名、版本号和报错信息。

    Args:
        query: 检索词或短句。
        k: 返回片段数，1-10。
    """
    if not query.strip():
        return []
    _, bm25 = _tools()
    result = bm25.search(query, k=_clamp(k))
    return [_hit_payload(hit) for hit in result.hits]


if __name__ == "__main__":
    mcp.run()
