"""Streamlit 薄客户端：不直接加载向量库或模型，只调用 FastAPI 服务。"""
import traceback
import uuid

import streamlit as st

from service_client import RagApiClient, RagApiError

st.set_page_config(page_title="FastAPI 文档问答", layout="wide")
st.title("FastAPI 知识库问答")


@st.cache_resource
def get_client() -> RagApiClient:
    return RagApiClient()


client = get_client()

if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"user_{uuid.uuid4().hex[:8]}"
if "display_messages" not in st.session_state:
    st.session_state["display_messages"] = []


def render_sources(sources):
    with st.expander(f"检索来源（{len(sources)} 条）"):
        for source in sources:
            st.markdown(
                f"**[{source['index']}] {source['source']} 第 {source['chunk_index']} 段** · "
                f"相关度 {source['score']}"
            )
            st.caption(source["snippet"])


with st.sidebar:
    st.caption(f"API：{client.base_url}")
    try:
        health = client.health()
        status_text = "正常" if health["status"] == "ok" else "降级"
        st.write(f"服务状态：{status_text}")
        st.caption(
            f"知识库 {health['checks']['chroma']['chunks']} 段 · "
            f"会话存储 {health['checks']['session']['backend']}"
        )
        if health["checks"]["session"].get("degraded"):
            st.warning("Redis 不可用，当前会话已降级为文件存储")
    except Exception as exc:
        st.error(f"无法连接问答服务：{exc}")
        st.caption("请先启动：uvicorn service.api:app --port 8000")
    if st.button("新会话"):
        st.session_state["session_id"] = f"user_{uuid.uuid4().hex[:8]}"
        st.session_state["display_messages"] = []
        st.rerun()

for message in st.session_state["display_messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])
        if message.get("caption"):
            st.caption(message["caption"])

prompt = st.chat_input("请输入问题")
if prompt:
    st.session_state["display_messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("正在检索…", expanded=False)
        final = None
        try:
            for event, data in client.stream(prompt, st.session_state["session_id"]):
                if event == "node":
                    label = f"{data['node']} · {data['ms']:.0f} ms"
                    if data.get("detail"):
                        label += f" · {data['detail'][:60]}"
                    status.update(label=label)
                elif event == "final":
                    final = data
                elif event == "error":
                    raise RagApiError(500, data.get("detail", "stream error"))
        except Exception as exc:
            status.update(label="问答失败", state="error", expanded=False)
            st.error(f"问答服务出错：{exc}")
            traceback.print_exc()

        if final:
            status.update(
                label=f"完成 · {final['metrics']['total_ms'] / 1000:.1f}s",
                state="complete",
                expanded=False,
            )
            st.markdown(final["answer"])
            caption_parts = []
            if final.get("query") and final["query"] != prompt:
                caption_parts.append(f"检索时改写为：{final['query']}")
            if final.get("refused"):
                caption_parts.append("检索结果不足以回答，已拒答")
            caption_parts.append(
                f"route={final.get('route')} · tool={final.get('tool_choice')} · "
                f"LLM {final['metrics']['llm_calls']} 次"
            )
            caption = "；".join(caption_parts)
            if final.get("verify") == "unsupported":
                st.warning("引用校验未通过：部分结论缺少资料支撑")
            if final.get("sources"):
                render_sources(final["sources"])
            st.caption(caption)
            st.session_state["display_messages"].append(
                {
                    "role": "assistant",
                    "content": final["answer"],
                    "sources": final.get("sources") or [],
                    "caption": caption,
                }
            )
