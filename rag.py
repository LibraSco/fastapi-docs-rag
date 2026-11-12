"""RAG 问答：检索、精排、多轮改写、引用与拒答。"""
import time
from dataclasses import dataclass, field

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import config_data as config
from file_history_store import get_history
from rerank import RerankService
from vector_stores import Hit, VectorStoreService

REFUSAL_TEXT = "根据现有资料无法回答该问题。"
REFUSAL_MARKERS = ("无法回答", "没有提到", "未提及", "资料中没有", "无法确定", "未包含", "无法提供")


@dataclass
class Answer:
    text: str
    sources: list[Hit] = field(default_factory=list)
    query: str = ""           # 实际用于检索的 query（多轮时是改写后的）
    refused: bool = False
    retrieval_ms: float = 0.0


class RagService(object):
    def __init__(self):
        config.require_api_key()
        self.vector_service = VectorStoreService()
        self.reranker = RerankService()
        self.llm = ChatTongyi(model=config.chat_model_name, dashscope_api_key=config.api_key)

        self.condense_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是检索查询改写器。结合对话历史，把用户最新问题改写成一个不依赖上下文、"
                    "可直接用于检索的完整问句。只输出改写后的问句，不要回答问题，不要解释；"
                    "如果问题本身已经完整，原样输出。",
                ),
                MessagesPlaceholder("history"),
                ("human", "最新问题：{input}"),
            ]
        )
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是知识库问答助手。只依据下面的参考资料回答，不要使用资料以外的知识。\n"
                    "规则：\n"
                    "1. 每个关键结论后用 [编号] 标注依据，编号只能来自参考资料。\n"
                    "2. 资料不足以回答时，直接回答“根据现有资料无法回答该问题。”，不要猜测或编造。\n"
                    "3. 回答简洁，先给结论，再给必要细节。\n\n"
                    "参考资料：\n{context}",
                ),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )

    def retrieve(self, query: str) -> list[Hit]:
        hits = self.vector_service.search(query, k=config.recall_k)
        return self.reranker.rerank(query, hits, top_n=config.final_k)

    def format_context(self, hits: list[Hit]) -> str:
        if not hits:
            return "（无检索结果）"
        blocks = [
            f"[{i}] 来源：{hit.source} 第 {hit.chunk_index} 段\n{hit.text}"
            for i, hit in enumerate(hits, 1)
        ]
        return "\n\n".join(blocks)

    def condense(self, question: str, history_messages: list) -> str:
        if not history_messages:
            return question
        try:
            messages = self.condense_prompt.format_messages(history=history_messages, input=question)
            rewritten = self.llm.invoke(messages).content.strip()
            rewritten = rewritten.splitlines()[0].strip().strip('"“”')
            return rewritten or question
        except Exception:
            return question

    def ask(self, question: str, session_id: str = None) -> Answer:
        session_id = session_id or config.default_session_id
        history = get_history(session_id)
        recent = history.messages[-config.max_history_messages:]
        standalone = self.condense(question, recent)

        start = time.perf_counter()
        hits = self.retrieve(standalone)
        retrieval_ms = (time.perf_counter() - start) * 1000

        if not hits:
            answer = Answer(REFUSAL_TEXT, [], standalone, True, retrieval_ms)
        else:
            messages = self.answer_prompt.format_messages(
                context=self.format_context(hits), history=recent, input=question
            )
            text = self.llm.invoke(messages).content.strip()
            refused = any(marker in text for marker in REFUSAL_MARKERS)
            answer = Answer(text, hits, standalone, refused, retrieval_ms)

        history.add_user_message(question)
        history.add_ai_message(answer.text)
        return answer
