"""向量检索：统一 Hit 结构，供精排、生成与评测复用。"""
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

import config_data as config


@dataclass
class Hit:
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    distance: float = 0.0        # 向量距离，越小越相近
    rerank_score: float | None = None   # 精排相关性分，越大越相关

    def label(self) -> str:
        return f"{self.source}#{self.chunk_index}"


class VectorStoreService(object):
    def __init__(self, embedding=None, collection_name=None):
        config.require_api_key()
        self.collection_name = collection_name or config.collection_name
        self.embedding = embedding or DashScopeEmbeddings(model=config.embedding_model_name)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def search(self, query: str, k: int = None) -> list:
        n = k or config.recall_k
        results = self.vector_store.similarity_search_with_score(query, k=n)
        hits = []
        for doc, distance in results:
            meta = doc.metadata or {}
            hits.append(
                Hit(
                    chunk_id=meta.get("chunk_id", ""),
                    text=doc.page_content,
                    source=meta.get("source", "unknown"),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    distance=float(distance),
                )
            )
        return hits
