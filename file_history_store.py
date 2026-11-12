"""基于文件的会话历史：按 session 存 JSON，写入时截断并原子替换。"""
import json
import logging
import os
import re
import tempfile
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

import config_data as config

logger = logging.getLogger(__name__)


def get_history(session_id: str) -> "FileChatMessageHistory":
    return FileChatMessageHistory(session_id, config.history_dir)


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, storage_path: str = None):
        if not re.fullmatch(r"[0-9A-Za-z_-]{1,64}", session_id):
            raise ValueError("session_id 只允许字母、数字、下划线和短横线")
        self.session_id = session_id
        self.storage_path = storage_path or config.history_dir
        os.makedirs(self.storage_path, exist_ok=True)
        self.file_path = os.path.join(self.storage_path, session_id)

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return messages_from_dict(json.load(f))
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            logger.warning("会话文件损坏，已按空历史处理: %s", self.file_path)
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        merged = list(self.messages) + list(messages)
        merged = merged[-config.max_history_messages:]
        self._write([message_to_dict(message) for message in merged])

    def clear(self) -> None:
        self._write([])

    def _write(self, payload: list) -> None:
        fd, tmp_path = tempfile.mkstemp(dir=self.storage_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_path, self.file_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
