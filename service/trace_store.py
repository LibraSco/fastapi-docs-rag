"""请求级 trace 暂存：进程内 TTL 存储，供 /trace/{request_id} 查询。

只保留最近一段时间的请求，重启即失效；需要跨实例排障时再接 Redis。
"""
import threading
import time
from collections import OrderedDict


class TraceStore:
    def __init__(self, ttl_seconds: int, max_items: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._items: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, request_id: str, payload: dict) -> None:
        now = time.time()
        with self._lock:
            self._prune(now)
            self._items[request_id] = (now + self.ttl_seconds, payload)
            self._items.move_to_end(request_id)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def get(self, request_id: str) -> dict | None:
        now = time.time()
        with self._lock:
            self._prune(now)
            item = self._items.get(request_id)
            return item[1] if item else None

    def _prune(self, now: float) -> None:
        while self._items:
            _, (expires_at, _) = next(iter(self._items.items()))
            if expires_at > now:
                break
            self._items.popitem(last=False)
