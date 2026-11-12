"""图节点：每个节点只做一件事，返回局部状态。

P1 的 route 固定走单跳，不调用模型；这样 50 题上的结果应当与旧 rag.py 的 C 组一致，
后面的路由、自检、拆解、校验每加一个都单独评测，避免混在一起无法归因。
"""
import re
import time
from typing import Optional

from langchain_core.messages import BaseMessage

import config_data as config
from rag import REFUSAL_MARKERS, REFUSAL_TEXT
from agent.prompts import (
    build_answer_prompt,
    build_condense_prompt,
    build_decompose_prompt,
    build_grade_prompt,
    build_regenerate_prompt,
    build_rewrite_prompt,
    build_route_prompt,
    build_verify_prompt,
)
from agent.state import TraceStep
from agent.tools import BM25Tool, VectorTool, dedupe_hits, source_anchor_hits

CITATION_RE = re.compile(r"\[(\d+)\]")


def format_context(hits) -> str:
    if not hits:
        return "（无检索结果）"
    blocks = [
        f"[{i}] 来源：{hit.source} 第 {hit.chunk_index} 段\n{hit.text}"
        for i, hit in enumerate(hits, 1)
    ]
    return "\n\n".join(blocks)


def parse_citations(text: str) -> list[int]:
    seen = []
    for raw in CITATION_RE.findall(text or ""):
        number = int(raw)
        if number not in seen:
            seen.append(number)
    return seen


def strip_invalid_citations(text: str, limit: int) -> str:
    """删掉越界的引用编号，保留正文；编号合法性与文档数绑定，不交给模型判断。"""

    def keep(match):
        number = int(match.group(1))
        return match.group(0) if 1 <= number <= limit else ""

    return CITATION_RE.sub(keep, text or "").strip()


REFUSAL_HEAD_RE = re.compile(
    r"^\s*(根据现有资料|根据提供的?资料|资料中没有|资料未提及|参考资料未提及|资料不足|无法回答|无法确定)"
)


def is_refusal_only(text: str) -> bool:
    """开头就是拒答口吻且没有任何引用，才算整题拒答；修正型回答里补充“资料未提及 X”不算。"""
    stripped = (text or "").strip()
    if not stripped:
        return True
    return bool(REFUSAL_HEAD_RE.match(stripped)) and not CITATION_RE.search(stripped)


def parse_label(text: str, labels: tuple, default: str) -> str:
    """标签解析：优先句首命中，其次整词命中；顺序由调用方决定（insufficient 要在 sufficient 前）。"""
    lowered = (text or "").strip().lower()
    for label in labels:
        if re.match(rf"^{label}\b", lowered):
            return label
    for label in labels:
        if re.search(rf"\b{label}\b", lowered):
            return label
    return default


def _usage(message) -> tuple[int, int]:
    meta = getattr(message, "usage_metadata", None) or {}
    if meta:
        return int(meta.get("input_tokens") or 0), int(meta.get("output_tokens") or 0)
    usage = (getattr(message, "response_metadata", None) or {}).get("token_usage") or {}
    return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)


class AgentNodes(object):
    def __init__(
        self,
        llm,
        vector_tool: VectorTool,
        bm25_tool: Optional[BM25Tool] = None,
        judge_llm=None,
        grade_skip_score: float = 0.75,
        bm25_skip_score: float = None,
    ):
        self.llm = llm
        self.judge_llm = judge_llm or llm
        self.vector = vector_tool
        self.bm25 = bm25_tool
        self.grade_skip_score = grade_skip_score
        self.bm25_skip_score = config.bm25_skip_score if bm25_skip_score is None else bm25_skip_score
        self.answer_prompt = build_answer_prompt()
        self.condense_prompt = build_condense_prompt()
        self.route_prompt = build_route_prompt()
        self.decompose_prompt = build_decompose_prompt()
        self.grade_prompt = build_grade_prompt()
        self.rewrite_prompt = build_rewrite_prompt()
        self.verify_prompt = build_verify_prompt()
        self.regenerate_prompt = build_regenerate_prompt()

    def condense(self, state) -> dict:
        start = time.perf_counter()
        history: list[BaseMessage] = state.get("history") or []
        question = state["question"]
        if not history:
            detail = "无历史，沿用原问题"
            return {
                "standalone": question,
                "search_query": question,
                "trace": [TraceStep("condense", (time.perf_counter() - start) * 1000, detail).as_dict()],
            }
        try:
            messages = self.condense_prompt.format_messages(history=history, input=question)
            response = self.llm.invoke(messages)
            rewritten = response.content.strip().splitlines()[0].strip().strip('"“”')
            rewritten = rewritten or question
            prompt_tokens, completion_tokens = _usage(response)
            return {
                "standalone": rewritten,
                "search_query": rewritten,
                "llm_calls": 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "trace": [
                    TraceStep(
                        "condense",
                        (time.perf_counter() - start) * 1000,
                        f"改写为：{rewritten}",
                    ).as_dict()
                ],
            }
        except Exception as exc:
            return {
                "standalone": question,
                "search_query": question,
                "trace": [
                    TraceStep("condense", (time.perf_counter() - start) * 1000, f"改写失败，沿用原问题：{exc}").as_dict()
                ],
            }

    def route(self, state) -> dict:
        start = time.perf_counter()
        query = state.get("standalone") or state["question"]
        try:
            response = self.judge_llm.invoke(self.route_prompt.format_messages(input=query))
            raw = (response.content or "").strip()
            route = parse_label(raw, ("answer", "multi", "single"), default="single")
            prompt_tokens, completion_tokens = _usage(response)
            first_line = (raw.splitlines() or [""])[0][:30]
            return {
                "route": route,
                "llm_calls": 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "trace": [
                    TraceStep("route", (time.perf_counter() - start) * 1000, f"{route}（{first_line}）").as_dict()
                ],
            }
        except Exception as exc:
            return {
                "route": "single",
                "trace": [
                    TraceStep("route", (time.perf_counter() - start) * 1000, f"路由失败，默认单跳：{exc}").as_dict()
                ],
            }

    def decompose(self, state) -> dict:
        start = time.perf_counter()
        query = state.get("standalone") or state["question"]
        try:
            response = self.judge_llm.invoke(self.decompose_prompt.format_messages(input=query))
            raw = (response.content or "").strip()
            sub_questions = []
            for line in raw.splitlines():
                cleaned = re.sub(r"^[\s\-*.\d、)）]+", "", line).strip().strip('"“”')
                if cleaned and cleaned not in sub_questions:
                    sub_questions.append(cleaned)
            sub_questions = sub_questions[:3] or [query]
            prompt_tokens, completion_tokens = _usage(response)
            return {
                "sub_questions": sub_questions,
                "llm_calls": 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "trace": [
                    TraceStep(
                        "decompose",
                        (time.perf_counter() - start) * 1000,
                        "；".join(sub_questions),
                    ).as_dict()
                ],
            }
        except Exception as exc:
            return {
                "sub_questions": [query],
                "trace": [
                    TraceStep("decompose", (time.perf_counter() - start) * 1000, f"拆解失败，按原问题检索：{exc}").as_dict()
                ],
            }

    def direct(self, state) -> dict:
        start = time.perf_counter()
        return {
            "answer": REFUSAL_TEXT,
            "refused": True,
            "hits": [],
            "citations": [],
            "trace": [
                TraceStep("direct", (time.perf_counter() - start) * 1000, "路由判定知识库无法回答，不检索直接拒答").as_dict()
            ],
        }

    @staticmethod
    def _inject_anchors(hits, anchors):
        """从尾部替换，保住头部精排顺序；已经是锚点的槽位不重复占用。"""
        chosen = list(hits)
        anchor_ids = {hit.chunk_id for hit in anchors}
        chosen_ids = {hit.chunk_id for hit in chosen}
        for anchor in anchors:
            if anchor.chunk_id in chosen_ids:
                continue
            slot = next(
                (index for index in range(len(chosen) - 1, -1, -1) if chosen[index].chunk_id not in anchor_ids),
                None,
            )
            if slot is None:
                continue
            chosen_ids.discard(chosen[slot].chunk_id)
            chosen[slot] = anchor
            chosen_ids.add(anchor.chunk_id)
        return chosen

    def _recall_queries(self, state) -> list[str]:
        query = state.get("search_query") or state.get("standalone") or state["question"]
        sub_questions = state.get("sub_questions") or []
        # 多跳时原问题也保留一路召回，拆解只做增量，避免误判多跳反而丢掉单跳能命中的片段
        if len(sub_questions) > 1:
            return [query] + [sub for sub in sub_questions if sub != query]
        return [query]

    def select_tool(self, state) -> dict:
        start = time.perf_counter()
        queries = self._recall_queries(state)
        if self.bm25 is None:
            choice, detail = "vector", "未配置 BM25，使用向量检索"
        else:
            scores = [self.bm25.top_score(query) for query in queries]
            lowest = min(scores) if scores else 0.0
            if scores and all(score >= self.bm25_skip_score for score in scores):
                choice = "bm25"
                detail = f"BM25 最低分 {lowest:.1f} ≥ {self.bm25_skip_score:.0f}，跳过向量检索"
            else:
                choice = "vector"
                detail = f"BM25 最低分 {lowest:.1f} < {self.bm25_skip_score:.0f}，使用向量+精排"
        return {
            "tool_choice": choice,
            "trace": [TraceStep("select_tool", (time.perf_counter() - start) * 1000, detail).as_dict()],
        }

    def retrieve(self, state) -> dict:
        start = time.perf_counter()
        query = state.get("search_query") or state.get("standalone") or state["question"]
        # 重查只换召回问法；精排始终围绕补全后的原问题，避免上一轮已覆盖的文档被改写句挤掉
        rerank_query = state.get("standalone") or state["question"]
        sub_questions = state.get("sub_questions") or [query]
        recall_queries = self._recall_queries(state)
        attempt = state.get("attempt", 0) + 1
        previous = state.get("candidates") or []
        tool_choice = state.get("tool_choice") or "vector"

        if tool_choice == "bm25" and self.bm25 is not None:
            groups = [self.bm25.search(item).hits for item in recall_queries]
            candidates = dedupe_hits([previous, *groups]) if previous else dedupe_hits(groups)
            hits = sorted(candidates, key=lambda hit: hit.distance)[: config.final_k]
            anchors = []
            if len(sub_questions) > 1 and hits and len({hit.source for hit in hits}) <= 2:
                anchors = source_anchor_hits(groups, hits)
                hits = self._inject_anchors(hits, anchors)
            top = hits[0] if hits else None
            detail = f"BM25 {' | '.join(recall_queries)} → 候选 {len(candidates)} → {len(hits)} 条"
            if top is not None:
                score = 1.0 / top.distance - 1.0 if top.distance > 0 else 0.0
                detail += f"（top1 {top.chunk_id[:8]} score={score:.1f}）"
            if anchors:
                detail += f"，补来源 {len(anchors)} 条"
        else:
            groups = [self.vector.recall(item).hits for item in recall_queries]
            # 重查只扩充候选池，不覆盖上一轮已找到的片段；多子查询按 chunk_id 去重后统一精排
            candidates = dedupe_hits([previous, *groups]) if previous else dedupe_hits(groups)
            ranked = self.vector.rerank(rerank_query, candidates)

            hits = list(ranked.hits)
            anchors = []
            # 只有精排结果集中在少数来源时才补来源，避免多跳兜底挤掉本来已命中的片段
            if len(sub_questions) > 1 and hits and len({hit.source for hit in hits}) <= 2:
                anchors = source_anchor_hits(groups, hits)
                hits = self._inject_anchors(hits, anchors)

            top = hits[0] if hits else None
            detail = f"{' | '.join(recall_queries)} → 候选 {len(candidates)} → {len(hits)} 条"
            if top is not None:
                detail += f"（top1 {top.chunk_id[:8]} score={top.rerank_score:.3f}）"
            if anchors:
                detail += f"，补来源 {len(anchors)} 条"

        update = {
            "hits": hits,
            "candidates": candidates,
            "queries": list(state.get("queries") or []) + list(recall_queries),
            "attempt": attempt,
            "tool_calls": len(recall_queries),
            "trace": [TraceStep("retrieve", (time.perf_counter() - start) * 1000, detail).as_dict()],
        }
        if attempt == 1:
            update["first_hits"] = hits
        return update

    def grade(self, state) -> dict:
        start = time.perf_counter()
        hits = state.get("hits") or []
        if not hits:
            return {
                "grade": "insufficient",
                "trace": [TraceStep("grade", (time.perf_counter() - start) * 1000, "无检索结果").as_dict()],
            }
        top = hits[0]
        if state.get("tool_choice") == "bm25":
            bm25_score = 1.0 / top.distance - 1.0 if top.distance > 0 else 0.0
            if bm25_score >= self.bm25_skip_score:
                return {
                    "grade": "sufficient",
                    "trace": [
                        TraceStep(
                            "grade",
                            (time.perf_counter() - start) * 1000,
                            f"BM25 {bm25_score:.1f} ≥ {self.bm25_skip_score:.0f}，跳过模型判定",
                        ).as_dict()
                    ],
                }
        elif top.rerank_score is not None and top.rerank_score >= self.grade_skip_score:
            return {
                "grade": "sufficient",
                "trace": [
                    TraceStep(
                        "grade",
                        (time.perf_counter() - start) * 1000,
                        f"top1 {top.rerank_score:.3f} ≥ {self.grade_skip_score:.2f}，跳过模型判定",
                    ).as_dict()
                ],
            }
        query = state.get("standalone") or state["question"]
        try:
            response = self.judge_llm.invoke(
                self.grade_prompt.format_messages(input=query, context=format_context(hits))
            )
            raw = (response.content or "").strip()
            label = parse_label(raw, ("insufficient", "sufficient"), default="sufficient")
            prompt_tokens, completion_tokens = _usage(response)
            return {
                "grade": label,
                "llm_calls": 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "trace": [
                    TraceStep("grade", (time.perf_counter() - start) * 1000, f"{label}（{raw[:20]}）").as_dict()
                ],
            }
        except Exception as exc:
            return {
                "grade": "sufficient",
                "trace": [
                    TraceStep("grade", (time.perf_counter() - start) * 1000, f"判定失败，默认放行：{exc}").as_dict()
                ],
            }

    def rewrite(self, state) -> dict:
        start = time.perf_counter()
        question = state.get("standalone") or state["question"]
        tried = state.get("queries") or []
        try:
            response = self.judge_llm.invoke(
                self.rewrite_prompt.format_messages(
                    input=question,
                    tried="；".join(tried),
                    reason=f"第 {state.get('attempt', 0)} 轮检索后仍未覆盖问题所需信息",
                )
            )
            raw = (response.content or "").strip()
            candidate = (raw.splitlines() or [""])[0].strip().strip('"“”')
            if not candidate or candidate in tried:
                candidate = f"{question} 用法 示例"
            prompt_tokens, completion_tokens = _usage(response)
            return {
                "search_query": candidate,
                "sub_questions": [],
                "llm_calls": 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "trace": [
                    TraceStep("rewrite", (time.perf_counter() - start) * 1000, f"改写为：{candidate}").as_dict()
                ],
            }
        except Exception as exc:
            fallback = f"{question} 用法 示例"
            return {
                "search_query": fallback,
                "sub_questions": [],
                "trace": [
                    TraceStep("rewrite", (time.perf_counter() - start) * 1000, f"改写失败，改用兜底查询：{exc}").as_dict()
                ],
            }

    def generate(self, state) -> dict:
        start = time.perf_counter()
        hits = state.get("hits") or []
        if not hits:
            return {
                "answer": REFUSAL_TEXT,
                "refused": True,
                "citations": [],
                "trace": [TraceStep("generate", (time.perf_counter() - start) * 1000, "无检索结果，直接拒答").as_dict()],
            }

        messages = self.answer_prompt.format_messages(
            context=format_context(hits),
            history=state.get("history") or [],
            input=state["question"],
        )
        response = self.llm.invoke(messages)
        text = response.content.strip()
        refused = (not text) or any(marker in text for marker in REFUSAL_MARKERS)
        citations = parse_citations(text)
        prompt_tokens, completion_tokens = _usage(response)
        return {
            "answer": text or REFUSAL_TEXT,
            "refused": refused,
            "citations": citations,
            "llm_calls": 1,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "trace": [
                TraceStep(
                    "generate",
                    (time.perf_counter() - start) * 1000,
                    f"{len(text)} 字，引用 {citations}，{'拒答' if refused else '正常回答'}",
                ).as_dict()
            ],
        }

    def verify(self, state) -> dict:
        start = time.perf_counter()
        hits = state.get("hits") or []
        text = state.get("answer") or ""
        attempt = (state.get("verify_attempt") or 0) + 1
        if state.get("refused") or not text or not hits:
            return {
                "verify": "skipped",
                "verify_attempt": attempt,
                "trace": [
                    TraceStep("verify", (time.perf_counter() - start) * 1000, "未生成答案或无检索结果，跳过校验").as_dict()
                ],
            }

        citations = parse_citations(text)
        invalid = sorted({n for n in citations if not 1 <= n <= len(hits)})
        cleaned = strip_invalid_citations(text, len(hits))
        update = {}
        if cleaned != text:
            update = {"answer": cleaned, "citations": [n for n in citations if 1 <= n <= len(hits)]}

        if invalid:
            verdict, reason = "unsupported", f"引用编号越界：{invalid}"
            detail = reason
        elif not citations:
            verdict, reason = "unsupported", "答案没有任何 [编号] 引用"
            detail = reason
        else:
            query = state.get("standalone") or state["question"]
            try:
                response = self.judge_llm.invoke(
                    self.verify_prompt.format_messages(
                        input=query,
                        context=format_context(hits),
                        answer=cleaned,
                    )
                )
                raw = (response.content or "").strip()
                verdict = parse_label(raw, ("unsupported", "pass"), default="pass")
                # 第一行是标签，后面的内容才是重生成时用得上的审核意见
                reason = "\n".join(raw.splitlines()[1:]).strip()[:200]
                prompt_tokens, completion_tokens = _usage(response)
                update["llm_calls"] = 1
                update["prompt_tokens"] = prompt_tokens
                update["completion_tokens"] = completion_tokens
                detail = f"{verdict}（{raw[:40]}）"
            except Exception as exc:
                verdict, reason = "pass", ""
                detail = f"校验失败，默认放行：{exc}"

        update.update(
            {
                "verify": verdict,
                "verify_attempt": attempt,
                "verify_reason": reason,
                "trace": [TraceStep("verify", (time.perf_counter() - start) * 1000, detail).as_dict()],
            }
        )
        return update

    def regenerate(self, state) -> dict:
        start = time.perf_counter()
        hits = state.get("hits") or []
        if not hits:
            return {
                "answer": REFUSAL_TEXT,
                "refused": True,
                "citations": [],
                "trace": [
                    TraceStep("regenerate", (time.perf_counter() - start) * 1000, "无检索结果，直接拒答").as_dict()
                ],
            }

        messages = self.regenerate_prompt.format_messages(
            context=format_context(hits),
            history=state.get("history") or [],
            previous=state.get("answer") or "",
            feedback=state.get("verify_reason") or "回答中有结论缺少资料支撑",
            input=state["question"],
        )
        try:
            response = self.llm.invoke(messages)
            text = (response.content or "").strip()
            refused = (not text) or is_refusal_only(text)
            citations = parse_citations(text)
            prompt_tokens, completion_tokens = _usage(response)
            return {
                "answer": text or REFUSAL_TEXT,
                "refused": refused,
                "citations": citations,
                "llm_calls": 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "trace": [
                    TraceStep(
                        "regenerate",
                        (time.perf_counter() - start) * 1000,
                        f"按审核意见重写，{len(text)} 字，引用 {citations}",
                    ).as_dict()
                ],
            }
        except Exception as exc:
            # 重生成失败时保留上一版答案，不因一次校验失败把可用结果丢掉
            return {
                "trace": [
                    TraceStep("regenerate", (time.perf_counter() - start) * 1000, f"重生成失败，保留上一版：{exc}").as_dict()
                ],
            }
