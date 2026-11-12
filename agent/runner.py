"""对外入口：组装依赖、执行图、维护会话历史。

会话存储由调用方注入（默认文件，服务层可换 Redis），保证多进程部署时行为一致。
"""
import time
from typing import Iterator, Optional

from langchain_community.chat_models.tongyi import ChatTongyi

import config_data as config
from agent.graph import build_graph
from agent.nodes import AgentNodes
from agent.sessions import SessionStore, build_session_store
from agent.state import AgentAnswer, initial_state
from agent.tools import BM25Tool, VectorTool


class AgentRagService(object):
    def __init__(
        self,
        llm=None,
        judge_llm=None,
        vector_tool: Optional[VectorTool] = None,
        bm25_tool=None,
        session_store: Optional[SessionStore] = None,
    ):
        config.require_api_key()
        self.vector = vector_tool or VectorTool()
        self.llm = llm or ChatTongyi(
            model=config.chat_model_name,
            dashscope_api_key=config.api_key,
        )
        # 路由/自检/改写要稳定标签，温度设 0；生成仍用默认温度，和旧链路可比
        self.judge_llm = judge_llm or (
            llm if llm is not None else ChatTongyi(
                model=config.chat_model_name,
                dashscope_api_key=config.api_key,
                temperature=0,
            )
        )
        self.bm25 = bm25_tool or BM25Tool(vector_service=self.vector.vector_service)
        self.nodes = AgentNodes(self.llm, self.vector, bm25_tool=self.bm25, judge_llm=self.judge_llm)
        self.graph = build_graph(self.nodes)
        self.sessions = session_store or build_session_store()

    def _history(self, session_id: Optional[str]) -> list:
        if not session_id:
            return []
        return self.sessions.get_messages(session_id)[-config.max_history_messages:]

    def _to_answer(self, final: dict, question: str, elapsed_ms: float) -> AgentAnswer:
        return AgentAnswer(
            text=final.get("answer") or "",
            sources=final.get("hits") or [],
            first_sources=final.get("first_hits") or [],
            query=final.get("standalone") or question,
            refused=bool(final.get("refused")),
            citations=final.get("citations") or [],
            trace=final.get("trace") or [],
            llm_calls=final.get("llm_calls") or 0,
            tool_calls=final.get("tool_calls") or 0,
            prompt_tokens=final.get("prompt_tokens") or 0,
            completion_tokens=final.get("completion_tokens") or 0,
            total_ms=elapsed_ms,
            route=final.get("route") or "",
            grade=final.get("grade") or "",
            verify=final.get("verify") or "",
            tool_choice=final.get("tool_choice") or "",
        )

    def ask(self, question: str, session_id: Optional[str] = None) -> AgentAnswer:
        history = self._history(session_id)
        start = time.perf_counter()
        final = self.graph.invoke(initial_state(question, history))
        elapsed = (time.perf_counter() - start) * 1000
        answer = self._to_answer(final, question, elapsed)
        if session_id:
            self.sessions.append_turn(session_id, question, answer.text)
        return answer

    def ask_stream(self, question: str, session_id: Optional[str] = None) -> Iterator[dict]:
        """按节点产出事件，最后一条是 final；供 SSE 使用。"""
        history = self._history(session_id)
        start = time.perf_counter()
        final = {}
        for mode, chunk in self.graph.stream(
            initial_state(question, history),
            stream_mode=["updates", "values"],
        ):
            if mode == "values":
                final = chunk
                continue
            for node, update in (chunk or {}).items():
                for step in (update or {}).get("trace") or []:
                    yield {"event": "node", "data": {"node": node, **step}}
        elapsed = (time.perf_counter() - start) * 1000
        answer = self._to_answer(final, question, elapsed)
        if session_id:
            self.sessions.append_turn(session_id, question, answer.text)
        yield {"event": "final", "data": answer.to_dict()}
