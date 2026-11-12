"""知识库入库：切分、稳定 chunk_id、按来源幂等重建。"""
import argparse
import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config


def make_chunk_id(source: str, chunk_index: int, text: str) -> str:
    # 同一来源、同一位置、同一内容 → ID 不变，重入库后评测结果可复现
    raw = f"{source}#{chunk_index}#{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


class KnowledgeBaseService(object):
    def __init__(self, collection_name=None):
        config.require_api_key()
        self.collection_name = collection_name or config.collection_name
        self.chroma = Chroma(
            collection_name=self.collection_name,
            embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),
            persist_directory=config.persist_directory,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def delete_by_source(self, source: str) -> int:
        existing = self.chroma.get(where={"source": source})["ids"]
        if existing:
            self.chroma.delete(ids=existing)
        return len(existing)

    def upload_by_str(self, data: str, filename: str, source: str = None) -> str:
        source = self._normalize_source(source or Path(filename).name)
        text = (data or "").strip()
        if not text:
            return "【跳过】内容为空"

        chunks = self.splitter.split_text(text)
        ids = [make_chunk_id(source, i, chunk) for i, chunk in enumerate(chunks)]
        metadatas = [
            {"source": source, "chunk_index": i, "chunk_id": chunk_id}
            for i, chunk_id in enumerate(ids)
        ]

        removed = self.delete_by_source(source)
        self.chroma.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
        return f"【成功】{source}：写入 {len(chunks)} 个片段，替换旧片段 {removed} 个"

    def count(self) -> int:
        return len(self.chroma.get(include=[])["ids"])

    @staticmethod
    def _normalize_source(name: str) -> str:
        parts = [part for part in name.replace("\\", "/").split("/") if part not in ("", ".")]
        if not parts or ".." in parts:
            raise ValueError(f"非法的来源名称: {name}")
        return "/".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="把单个文本文件写入知识库")
    parser.add_argument("path", help="TXT / Markdown 文件路径")
    args = parser.parse_args()

    file_path = Path(args.path)
    service = KnowledgeBaseService()
    print(service.upload_by_str(file_path.read_text(encoding="utf-8"), file_path.name))
    print("当前库内片段总数:", service.count())
