"""精排：用 gte-rerank 对向量召回结果重新打分排序。"""
import logging

import dashscope

import config_data as config
from vector_stores import Hit

logger = logging.getLogger(__name__)


class RerankService(object):
    def __init__(self, model: str = None):
        config.require_api_key()
        self.model = model or config.rerank_model_name

    def rerank(self, query: str, hits: list[Hit], top_n: int = None, fallback: bool = True) -> list[Hit]:
        if not hits:
            return []
        n = min(top_n or config.final_k, len(hits))
        try:
            resp = dashscope.TextReRank.call(
                model=self.model,
                query=query,
                documents=[hit.text for hit in hits],
                top_n=n,
                return_documents=False,
                api_key=config.api_key,
            )
        except Exception as exc:
            if not fallback:
                raise
            logger.warning("rerank 调用异常，降级为向量排序: %s", exc)
            return hits[:n]

        if resp.status_code != 200:
            if not fallback:
                raise RuntimeError(f"rerank 失败: {resp.status_code} {resp.message}")
            logger.warning("rerank 失败（%s），降级为向量排序", resp.message)
            return hits[:n]

        reranked = []
        for item in resp.output["results"]:
            hit = hits[item["index"]]
            hit.rerank_score = float(item["relevance_score"])
            reranked.append(hit)
        return reranked
