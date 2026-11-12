"""会话存储：文件（默认）与 Redis 两种实现，接口一致，供服务层切换。

文件实现沿用 file_history_store，保证和旧链路同一份行为；Redis 用来支撑多进程部署。
Redis 连不上时自动降级为文件，降级原因只记录异常类型，不把连接串带进日志或接口。
"""
import json
import logging
import re
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

import config_data as config
from file_history_store import get_history

logger = logging.getLogger(__name__)
SESSION_ID_RE = re.compile(r"[0-9A-Za-z_-]{1,64}")


def validate_session_id(session_id: str) -> str:
    if not SESSION_ID_RE.fullmatch(session_id or ""):
        raise ValueError("session_id 只允许字母、数字、下划线和短横线，长度 1-64")
    return session_id


class SessionStore(Protocol):
    backend: str
    degraded: str

    def get_messages(self, session_id: str) -> list[BaseMessage]: ...

    def append_turn(self, session_id: str, question: str, answer: str) -> None: ...


class FileSessionStore:
    backend = "file"

    def __init__(self, degraded: str = ""):
        self.degraded = degraded

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        return get_history(validate_session_id(session_id)).messages

    def append_turn(self, session_id: str, question: str, answer: str) -> None:
        history = get_history(validate_session_id(session_id))
        history.add_user_message(question)
        history.add_ai_message(answer)


class RedisSessionStore:
    backend = "redis"

    def __init__(self, url: str, ttl_seconds: int, max_messages: int, client=None):
        if client is None:
            import redis

            client = redis.Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
        self.client = client
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages
        self.degraded = ""

    def _key(self, session_id: str) -> str:
        return f"rag:session:{session_id}"

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        key = self._key(validate_session_id(session_id))
        messages = []
        for raw in self.client.lrange(key, 0, -1):
            try:
                item = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            content = item.get("content") or ""
            if item.get("role") == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        return messages[-self.max_messages :]

    def append_turn(self, session_id: str, question: str, answer: str) -> None:
        key = self._key(validate_session_id(session_id))
        pipe = self.client.pipeline()
        pipe.rpush(
            key,
            json.dumps({"role": "user", "content": question}, ensure_ascii=False),
            json.dumps({"role": "assistant", "content": answer}, ensure_ascii=False),
        )
        pipe.ltrim(key, -self.max_messages, -1)
        pipe.expire(key, self.ttl_seconds)
        pipe.execute()


def build_session_store() -> SessionStore:
    if config.session_backend != "redis":
        return FileSessionStore()
    try:
        store = RedisSessionStore(
            config.redis_url,
            config.session_ttl_seconds,
            config.max_history_messages,
        )
        store.client.ping()
        return store
    except Exception as exc:
        logger.warning("Redis 不可用，会话降级为文件存储：%s", type(exc).__name__)
        return FileSessionStore(degraded=type(exc).__name__)
