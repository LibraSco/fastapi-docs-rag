"""FastAPI 服务的 HTTP 客户端：Streamlit 只通过它访问问答链路。"""
import json
import os

import requests

import config_data as config


class RagApiError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class RagApiClient:
    def __init__(self, base_url: str = None, api_key: str = None, timeout: float = None):
        self.base_url = (base_url or os.getenv("RAG_API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.api_key = config.service_api_key if api_key is None else api_key
        self.timeout = timeout or float(os.getenv("RAG_API_TIMEOUT", "90"))

    def _headers(self) -> dict:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    @staticmethod
    def _raise_for_status(response) -> None:
        if response.ok:
            return
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RagApiError(response.status_code, str(detail)[:200])

    def health(self) -> dict:
        response = requests.get(f"{self.base_url}/health", headers=self._headers(), timeout=3)
        self._raise_for_status(response)
        return response.json()

    def ask(self, question: str, session_id: str = None) -> dict:
        response = requests.post(
            f"{self.base_url}/chat",
            json={"question": question, "session_id": session_id},
            headers=self._headers(),
            timeout=self.timeout,
        )
        self._raise_for_status(response)
        return response.json()

    def stream(self, question: str, session_id: str = None):
        """逐个产出 (event, data)，包括 start / node / final / error。"""
        payload = {"question": question, "session_id": session_id}
        with requests.post(
            f"{self.base_url}/chat/stream",
            json=payload,
            headers=self._headers(),
            stream=True,
            timeout=(5, self.timeout),
        ) as response:
            self._raise_for_status(response)
            event = None
            for raw in response.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if raw.startswith("event: "):
                    event = raw[len("event: ") :].strip()
                elif raw.startswith("data: "):
                    yield event, json.loads(raw[len("data: ") :])
