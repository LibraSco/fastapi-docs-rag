"""进程内限流：滑动窗口，按调用方 key 计数。

单实例够用；多副本部署时每个副本各算一份，需要全局限流再把计数换 Redis。
"""
import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0, max_keys: int = 10000):
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._buckets: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets[key]
            while bucket and bucket[0] <= now - self.window_seconds:
                bucket.popleft()
            if len(bucket) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - bucket[0])) + 1)
                return False, retry_after
            bucket.append(now)
            if len(self._buckets) > self.max_keys:
                self._drop_idle(now)
            return True, 0

    def _drop_idle(self, now: float) -> None:
        cutoff = now - self.window_seconds
        for key in [k for k, bucket in self._buckets.items() if not bucket or bucket[-1] <= cutoff]:
            self._buckets.pop(key, None)
