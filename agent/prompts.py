"""提示词集中管理。

答案提示与多轮改写提示先与 rag.py 保持逐字一致，P1 才能复现 C 组行为；
路由、自检、引用校验的提示在后续阶段追加，改动都有独立评测对照。
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ANSWER_SYSTEM = (
    "你是知识库问答助手。只依据下面的参考资料回答，不要使用资料以外的知识。\n"
    "规则：\n"
    "1. 每个关键结论后用 [编号] 标注依据，编号只能来自参考资料。\n"
    "2. 资料不足以回答时，直接回答“根据现有资料无法回答该问题。”，不要猜测或编造。\n"
    "3. 回答简洁，先给结论，再给必要细节。\n\n"
    "参考资料：\n{context}"
)

CONDENSE_SYSTEM = (
    "你是检索查询改写器。结合对话历史，把用户最新问题改写成一个不依赖上下文、"
    "可直接用于检索的完整问句。只输出改写后的问句，不要回答问题，不要解释；"
    "如果问题本身已经完整，原样输出。"
)

ROUTE_SYSTEM = (
    "你是检索路由器。知识库是 FastAPI 中文文档，也包含教程涉及的生态内容："
    "Pydantic、SQLModel、JWT 与安全、依赖注入、测试、部署（Docker、代理、HTTPS）、版本升级等。\n"
    "请判断用户问题属于哪一类，只输出一个英文标签，不要解释：\n"
    "- answer：只有明显与 FastAPI 及其生态无关时才选，例如纯闲聊、情绪表达，"
    "或明确问其他框架/语言（Django、Flask、Vue、Java 等）\n"
    "- single：一个具体的 FastAPI 或其生态问题，一次检索足够\n"
    "- multi：需要同时参考两篇或以上文档才能回答的 FastAPI 问题。"
    "常见信号是句子里有两个不同的机制或概念，需要都答到才算完整，"
    "例如“A 和 B 怎么配合”“既要做 X 又要做 Y”“谁先执行”“为什么 A 还要 B”；"
    "前后半句各问了一件事，也选 multi\n\n"
    "注意：口语化、带个人经历或抱怨、但主题仍在 FastAPI 生态内的问题，属于 single 或 multi，不要选 answer。"
    "部署平台与云服务、容器与代理、CPU/内存/异步阻塞等性能问题、依赖与版本升级，也都属于域内。"
    "拿不准时选 single。\n\n"
    "示例：\n"
    "问题：路径参数怎么做类型校验？ → single\n"
    "问题：怎么给一组接口统一加鉴权，又能让单个接口单独覆盖？ → multi\n"
    "问题：我这接口上线就 502，是不是代理配错了？ → single\n"
    "问题：Django 的 ORM 怎么批量更新？ → answer\n"
    "问题：今天好累，随便聊聊 → answer"
)

GRADE_SYSTEM = (
    "你是检索结果审核员。判断下面给出的资料是否足以回答问题。\n"
    "规则：\n"
    "1. 资料必须直接包含回答所需的信息，不能靠常识或资料外的知识补全。\n"
    "2. 问题包含多个部分时，资料必须覆盖每个部分。\n"
    "3. 只输出 sufficient 或 insufficient，不要解释。"
)

DECOMPOSE_SYSTEM = (
    "你是 FastAPI 知识库的检索查询规划器。用户问题需要同时参考两篇或以上文档才能答全。\n"
    "请把它拆成 2 个可以独立检索的子问题，每个覆盖原问题的一个信息需求。\n"
    "子问题必须围绕 FastAPI 及其生态提问，不要换成其他框架。\n"
    "规则：\n"
    "1. 每个子问题都要能脱离上下文单独使用，不要出现“它”“这个”“上面”等指代。\n"
    "2. 每行只输出一个子问题，不要编号，不要解释，不要直接回答问题。\n"
    "3. 如果问题本身无法拆分，原样输出。"
)

REWRITE_SYSTEM = (
    "你是检索查询改写器。上一轮检索没有找到足够资料。\n"
    "结合原问题和已经尝试过的查询，换一个更具体、更接近官方文档术语的问法；"
    "如果原问题包含多个部分，保留最可能需要补查的那部分。\n"
    "只输出新的检索问句，不要回答问题，不要解释。"
)

VERIFY_SYSTEM = (
    "你是引用审核员。根据给出的参考资料，检查回答是否忠实。\n"
    "规则：\n"
    "1. 回答里每个带 [编号] 的结论，编号必须在参考资料范围内。\n"
    "2. 被引用的资料必须真的支持这条结论：允许同义改写、把文档示例里的变量换成具体值；"
    "但不能靠常识补全，也不能把一篇文档的结论记到另一篇头上。\n"
    "3. 回答完全没有引用，或存在没有资料支撑的结论，判 unsupported。\n"
    "第一行只输出 pass 或 unsupported，不要解释；"
    "判 unsupported 时，第二行起逐条抄出没有资料支撑的那句结论（最多 3 条），"
    "不要写“根据现有资料无法回答”，也不要给修改建议。"
)

REGENERATE_HUMAN = (
    "上一版回答：\n{previous}\n\n"
    "审核意见（以下结论缺少资料支撑）：\n{feedback}\n\n"
    "请只依据参考资料重新回答原问题，每个关键结论都要有 [编号] 支撑。\n"
    "如果资料只能支撑部分结论，就只回答有支撑的部分，并说明哪部分资料没有覆盖，不要整题拒答；"
    "只有完全找不到依据时，才回答“根据现有资料无法回答该问题。”。\n\n"
    "原问题：{input}"
)


def build_answer_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )


def build_condense_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", CONDENSE_SYSTEM),
            MessagesPlaceholder("history"),
            ("human", "最新问题：{input}"),
        ]
    )


def build_route_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", ROUTE_SYSTEM), ("human", "问题：{input}")]
    )


def build_grade_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", GRADE_SYSTEM),
            ("human", "问题：{input}\n\n资料：\n{context}"),
        ]
    )


def build_decompose_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", DECOMPOSE_SYSTEM), ("human", "问题：{input}")]
    )


def build_rewrite_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", REWRITE_SYSTEM),
            ("human", "原问题：{input}\n已尝试的查询：{tried}\n上一轮资料为什么不够：{reason}"),
        ]
    )


def build_verify_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", VERIFY_SYSTEM),
            ("human", "问题：{input}\n\n参考资料：\n{context}\n\n回答：\n{answer}"),
        ]
    )


def build_regenerate_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM),
            MessagesPlaceholder("history"),
            ("human", REGENERATE_HUMAN),
        ]
    )
