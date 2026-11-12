"""集中配置：路径基于文件位置解析，模型与密钥从环境变量读取。"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
# override=True：项目 .env 优先于系统里残留的旧变量
load_dotenv(BASE_DIR / ".env", override=True)

api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

chat_model_name = os.getenv("CHAT_MODEL", "qwen-plus")
embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
rerank_model_name = os.getenv("RERANK_MODEL", "gte-rerank-v2")

collection_name = "rag"
persist_directory = str(BASE_DIR / "chroma_db")
history_dir = str(BASE_DIR / "chat_history")

chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " "]

# 两阶段检索：先召回 recall_k 条，精排后取 final_k 条
recall_k = 20
final_k = 5

# BM25 最高分达到该值就走词法检索，跳过向量召回与精排；阈值在 50 题评测集上校准
bm25_skip_score = 30.0

max_history_messages = 12
default_session_id = "user_001"

# 服务层：默认单机文件会话；配置 REDIS_URL 后切 Redis，连不上时自动降级
session_backend = os.getenv("SESSION_BACKEND", "redis" if os.getenv("REDIS_URL") else "file").strip().lower()
redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0").strip()
session_ttl_seconds = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
service_api_key = os.getenv("SERVICE_API_KEY", "").strip()
rate_limit_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
chat_timeout_seconds = float(os.getenv("CHAT_TIMEOUT_SECONDS", "60"))
trace_ttl_seconds = int(os.getenv("TRACE_TTL_SECONDS", "600"))


def require_api_key() -> str:
    if not api_key:
        raise RuntimeError("未检测到 DASHSCOPE_API_KEY，请在项目根目录 .env 中配置")
    return api_key
