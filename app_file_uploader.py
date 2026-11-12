import hashlib

import streamlit as st

from knowledge_base import KnowledgeBaseService

st.title("知识库更新")
st.caption("支持 TXT / Markdown，UTF-8 或 GB18030 编码。重复上传同一文件不会产生重复片段。")

uploaded = st.file_uploader(
    label="上传文本文件",
    type=["txt", "md", "markdown"],
    accept_multiple_files=False,
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploaded is not None:
    raw = uploaded.getvalue()
    digest = hashlib.md5(raw).hexdigest()

    if st.session_state.get("last_upload") == digest:
        st.info("该文件已入库，内容未变化")
    else:
        text = None
        for encoding in ("utf-8", "gb18030"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            st.error("无法解码文件，请转成 UTF-8 后重试")
        else:
            result = st.session_state["service"].upload_by_str(text, uploaded.name)
            st.session_state["last_upload"] = digest
            st.success(result)

    st.caption(
        f"文件：{uploaded.name} · {uploaded.size / 1024:.1f} KB · "
        f"当前库内片段 {st.session_state['service'].count()} 个"
    )
else:
    st.caption(f"当前库内片段 {st.session_state['service'].count()} 个")
