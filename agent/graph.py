"""LangGraph 装配：节点注册与边。

流程：
    condense → route ─┬─→ direct（知识库答不了，直接拒答）
                      ├─→ select_tool → retrieve → grade ─┬─→ generate → verify ─┬─→ END
                      │                                   │                     └─→ regenerate → verify（最多 1 次）
                      │                                   └─→ rewrite → retrieve（最多 2 轮）
                      └─→ decompose → select_tool → retrieve（多跳问题拆成子问题）
"""
from langgraph.graph import END, START, StateGraph

from agent.nodes import AgentNodes
from agent.state import AgentState

MAX_RETRIEVAL_ROUNDS = 2
MAX_VERIFY_ROUNDS = 2


def next_after_route(state: AgentState) -> str:
    route = state.get("route") or "single"
    if route == "multi":
        return "decompose"
    return "select_tool" if route == "single" else "direct"


def next_after_grade(state: AgentState) -> str:
    if state.get("grade") == "sufficient":
        return "generate"
    if (state.get("attempt") or 0) >= MAX_RETRIEVAL_ROUNDS:
        return "generate"
    return "rewrite"


def next_after_verify(state: AgentState) -> str:
    if state.get("verify") == "unsupported" and (state.get("verify_attempt") or 0) < MAX_VERIFY_ROUNDS:
        return "regenerate"
    return "end"


def build_graph(nodes: AgentNodes):
    builder = StateGraph(AgentState)
    builder.add_node("condense", nodes.condense)
    builder.add_node("route", nodes.route)
    builder.add_node("decompose", nodes.decompose)
    builder.add_node("select_tool", nodes.select_tool)
    builder.add_node("direct", nodes.direct)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("grade", nodes.grade)
    builder.add_node("rewrite", nodes.rewrite)
    builder.add_node("generate", nodes.generate)
    builder.add_node("verify", nodes.verify)
    builder.add_node("regenerate", nodes.regenerate)

    builder.add_edge(START, "condense")
    builder.add_edge("condense", "route")
    builder.add_conditional_edges(
        "route",
        next_after_route,
        {"select_tool": "select_tool", "decompose": "decompose", "direct": "direct"},
    )
    builder.add_edge("decompose", "select_tool")
    builder.add_edge("select_tool", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        next_after_grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", "verify")
    builder.add_conditional_edges(
        "verify",
        next_after_verify,
        {"regenerate": "regenerate", "end": END},
    )
    builder.add_edge("regenerate", "verify")
    builder.add_edge("direct", END)
    return builder.compile()
