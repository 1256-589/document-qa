import streamlit as st
from PyPDF2 import PdfReader
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chat_models import init_chat_model
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
import os
from dotenv import load_dotenv
import shutil
import time
import pickle
from datetime import datetime

# --- 初始化与配置 ---
load_dotenv(override=True)
st.set_page_config(page_title="AI 研究助手", page_icon="💡")

# --- 常量定义 ---
DB_DIR = "document_database"
FILES_DIR = os.path.join(DB_DIR, "files")
INDEX_PATH = os.path.join(DB_DIR, "faiss_index")
CHAT_HISTORY_DIR = "chat_history"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --- 后端辅助函数 ---
@st.cache_resource
def get_embeddings():
    return DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))


embeddings = get_embeddings()


def setup_directories():
    os.makedirs(FILES_DIR, exist_ok=True)
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)


# --- 数据存储函数 (重大修改) ---
def save_chat_history(title, messages, file_path):
    """保存对话标题和消息到 Pickle 文件"""
    if not file_path:
        st.error("错误：没有提供有效的对话文件路径用于保存！")
        return
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump({"title": title, "messages": messages}, f)


def load_chat_history(file_path):
    """从 Pickle 文件加载对话，返回 (标题, 消息列表)"""
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                # 兼容新旧两种数据格式
                if isinstance(data, dict):
                    return data.get("title"), data.get("messages", [])
                elif isinstance(data, list):
                    return None, data  # 旧格式，没有标题
        except (pickle.UnpicklingError, EOFError):
            return "损坏的对话", [AIMessage(content="无法加载对话，文件可能已损坏。")]
    return None, []


def get_chat_title_from_file(file_path):
    """只读取标题，用于在侧边栏快速显示，避免加载整个对话历史"""
    title, _ = load_chat_history(file_path)
    return title


# --- LLM 相关函数 ---
def generate_chat_title(query):
    """调用LLM为对话生成一个简短的标题"""
    st.toast("正在为新对话生成标题...", icon="🏷️")
    title_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个文本摘要助手。请根据用户的提问，为这个对话生成一个不超过8个字的、简短精炼的标题。"),
        ("user", "{query}")
    ])
    # 为了节省成本和提高速度，可以使用一个较小的模型
    llm = init_chat_model("deepseek-chat", model_provider="deepseek", temperature=0.1)
    chain = title_prompt | llm
    try:
        response = chain.invoke({"query": query})
        return response.content.strip().replace("\"", "")
    except Exception as e:
        print(f"标题生成失败: {e}")
        return "新对话"


# ... (process_and_index_files, get_relevant_files, get_agent_executor, run_agent_with_streaming 函数保持不变) ...
def process_and_index_files(pdf_docs, progress_placeholder):
    """处理上传的PDF文件，保存、切分并创建或更新向量索引"""
    texts_with_metadata = []
    progress_bar = progress_placeholder.progress(0, text="开始处理文件...")

    for i, pdf in enumerate(pdf_docs):
        file_path = os.path.join(FILES_DIR, pdf.name)
        with open(file_path, "wb") as f:
            f.write(pdf.getbuffer())

        pdf_reader = PdfReader(file_path)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(text)
                for chunk in chunks:
                    texts_with_metadata.append((chunk, {"source": pdf.name}))

        progress_bar.progress((i + 1) / len(pdf_docs), text=f"正在处理: {pdf.name}")

    if not texts_with_metadata:
        progress_bar.empty()
        return 0

    texts = [item[0] for item in texts_with_metadata]
    metadatas = [item[1] for item in texts_with_metadata]

    if os.path.exists(INDEX_PATH):
        db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        db.add_texts(texts, metadatas=metadatas)
    else:
        db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    db.save_local(INDEX_PATH)
    progress_bar.progress(1.0, text="所有文件处理完成！")
    return len(pdf_docs)


def get_relevant_files(query):
    """根据查询在整个数据库中检索相关文件名"""
    if not os.path.exists(INDEX_PATH):
        return []

    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 20})
    relevant_docs = retriever.get_relevant_documents(query)

    return list(set(doc.metadata['source'] for doc in relevant_docs))


def get_agent_executor(selected_files=None):
    """创建带有工具的 Agent Executor。如果选择了文件，则文档检索器会被过滤"""
    search = TavilySearchResults(max_results=6)

    @tool
    def search_tool(query: str) -> str:
        """当你需要搜索最新信息或用户问题与本地文档无关时使用此工具。"""
        return search.invoke(query)

    tools = [search_tool]

    if selected_files and os.path.exists(INDEX_PATH):
        full_db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

        filtered_docs = []
        for doc_id, doc in full_db.docstore._dict.items():
            if doc.metadata.get("source") in selected_files:
                filtered_docs.append(doc)

        if filtered_docs:
            filtered_db = FAISS.from_documents(filtered_docs, embeddings)
            retriever = filtered_db.as_retriever(search_kwargs={'k': 10})  # 增加检索数量

            document_retriever_tool = create_retriever_tool(
                retriever,
                "document_retriever",
                f"优先使用此工具回答关于特定文档的问题。这些是用户授权你查阅的文档：{', '.join(selected_files)}。"
            )
            tools.append(document_retriever_tool)

    llm = init_chat_model("deepseek-chat", model_provider="deepseek", temperature=0.7)
    prompt = hub.pull("hwchase17/openai-tools-agent")

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


def run_agent_with_streaming(agent_executor, query, history):
    """运行 Agent 并通过回调实现流式输出"""

    class StreamingCallbackHandler(BaseCallbackHandler):
        def __init__(self, placeholder):
            self.placeholder = placeholder
            self.full_response = ""

        def on_llm_new_token(self, token: str, **kwargs: any) -> None:
            self.full_response += token
            self.placeholder.markdown(self.full_response + "▌")  # 使用光标模拟打字效果

        def on_agent_action(self, action: AgentAction, **kwargs: any) -> any:
            tool_name = action.tool
            self.placeholder.markdown(f"正在调用工具: `{tool_name}`...")

    placeholder = st.empty()
    streaming_handler = StreamingCallbackHandler(placeholder)

    response = agent_executor.invoke(
        {"input": query, "chat_history": history},
        config={"callbacks": [streaming_handler]}
    )

    output = response.get('output', '处理出错，请重试。')
    placeholder.markdown(output)
    return output


# --- 主程序界面 ---
def start_new_chat():
    """创建一个新的对话，并立即持久化"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_chat_filename = f"chat_{timestamp}.pkl"
    st.session_state.active_chat_file_path = os.path.join(CHAT_HISTORY_DIR, new_chat_filename)

    st.session_state.active_chat_title = None
    st.session_state.chat_history = [AIMessage(content="您好！新的对话已开启。")]
    st.session_state.file_selection = {"show": False, "query": None, "options": []}

    save_chat_history(
        st.session_state.active_chat_title,
        st.session_state.chat_history,
        st.session_state.active_chat_file_path
    )
    st.rerun()


def main():
    setup_directories()

    def load_css(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    load_css('style.css')

    # --- 会话状态初始化 ---
    if "chat_history" not in st.session_state:
        chat_files = sorted(
            [f for f in os.listdir(CHAT_HISTORY_DIR) if f.startswith("chat_") and f.endswith(".pkl")],
            reverse=True
        )
        if chat_files:
            latest_chat_file = chat_files[0]
            st.session_state.active_chat_file_path = os.path.join(CHAT_HISTORY_DIR, latest_chat_file)
            title, messages = load_chat_history(st.session_state.active_chat_file_path)
            st.session_state.active_chat_title = title
            st.session_state.chat_history = messages
        else:
            # 首次运行时创建第一个对话
            start_new_chat()
            # st.rerun() 会导致 Streamlit 重新执行，所以这里不需要再写代码

    # --- 侧边栏 UI ---
    with st.sidebar:
        # st.divider()

        u1, u2 = st.columns([0.7, 0.3])
        with u1:
            st.title("RAG Agent")
        with u2:
            st.session_state.rag_active = st.toggle("RAG", value=False,label_visibility="collapsed")

        # 当对话历史>1条时（即用户已提问），才允许新建对话
        is_chatting = len(st.session_state.get("chat_history", [])) > 1
        if st.button("📝 新建对话", use_container_width=True, disabled=not is_chatting,type="primary"):
            start_new_chat()
        st.divider()
        st.subheader("🗓️ 对话历史")
        all_chats = sorted(
            [f for f in os.listdir(CHAT_HISTORY_DIR) if f.startswith("chat_") and f.endswith(".pkl")],
            reverse=True
        )
        st.divider()

        for chat_file in all_chats:
            file_path = os.path.join(CHAT_HISTORY_DIR, chat_file)
            title = get_chat_title_from_file(file_path)

            if not title:
                ts_str = chat_file.replace("chat_", "").replace(".pkl", "")
                try:
                    title = f"对话于 {datetime.strptime(ts_str, '%Y%m%d_%H%M%S').strftime('%y-%m-%d %H:%M')}"
                except ValueError:
                    title = "未知对话"

            is_active = file_path == st.session_state.get("active_chat_file_path")
            label = f"**{title}**" if is_active else title

            if st.button(label, key=chat_file, use_container_width=True):
                st.session_state.active_chat_file_path = file_path
                title, messages = load_chat_history(file_path)
                st.session_state.active_chat_title = title
                st.session_state.chat_history = messages
                st.rerun()
        st.divider()

        pdf_docs = st.file_uploader("上传新的 PDF 文档", accept_multiple_files=True, type=['pdf'],label_visibility="collapsed")

        if st.button("▶️ 开始处理", disabled=not pdf_docs, use_container_width=True,type="primary"):
            progress_placeholder = st.empty()
            with st.spinner("正在处理和索引文档..."):
                count = process_and_index_files(pdf_docs, progress_placeholder)
            st.success(f"成功处理并索引了 {count} 个新文件。")
            time.sleep(2)
            progress_placeholder.empty()
            st.rerun()
        st.divider()

        with st.expander("📚 文档库", expanded=True):

            indexed_files = [f for f in os.listdir(FILES_DIR) if f.endswith('.pdf')] if os.path.exists(
                    FILES_DIR) else []
            if indexed_files:
                st.caption(f"当前库中有 {len(indexed_files)} 个文档")
                for file_name in indexed_files:
                    st.markdown(f"📄 {file_name}")
            else:
                st.info("文档库为空。")

        st.divider()
        if st.button("🗑️ 清空所有数据", use_container_width=True,type="primary"):
            # ... UI无变化
            # if st.checkbox("确认要删除所有文档和聊天记录吗？"):
            if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)
            if os.path.exists(CHAT_HISTORY_DIR): shutil.rmtree(CHAT_HISTORY_DIR)
            st.session_state.clear()
            st.success("所有数据已被清空。应用将在一秒后刷新。")
            time.sleep(1)
            st.rerun()

    # --- 主聊天界面 ---
    for message in st.session_state.get("chat_history", []):
        avatar = "💡" if isinstance(message, AIMessage) else "🧑‍💻"
        with st.chat_message(message.type, avatar=avatar):
            st.write(message.content)

    if st.session_state.get("file_selection", {}).get("show"):
        # ... RAG选择界面无变化
        with st.chat_message("ai", avatar="💡"):
            st.info("根据您的问题，我找到了这些可能相关的文档。请选择您希望我重点查阅的文件：")

            selected_files = st.multiselect(
                "选择文档:",
                options=st.session_state.file_selection["options"],
                default=st.session_state.file_selection["options"],
                label_visibility="collapsed"
            )

            if st.button("基于所选文件生成回答", type="primary"):
                query = st.session_state.file_selection["query"]
                # st.session_state.chat_history.append(HumanMessage(content=query))

                with st.chat_message("user", avatar="🧑‍💻"):
                    st.write(query)

                with st.chat_message("ai", avatar="💡"):
                    with st.spinner("正在基于文档思考..."):
                        agent_executor = get_agent_executor(selected_files)
                        final_query = f"请严格且优先基于提供的 `document_retriever` 工具来回答以下问题。问题是：{query}"
                        response_content = run_agent_with_streaming(agent_executor, final_query,
                                                                    st.session_state.chat_history[1:-1])

                st.session_state.chat_history.append(AIMessage(content=response_content))
                # 实时保存到当前激活的对话文件中
                save_chat_history(st.session_state.active_chat_title, st.session_state.chat_history,
                                  st.session_state.active_chat_file_path)
                st.session_state.file_selection["show"] = False
                st.rerun()

    if user_query := st.chat_input("请输入您的问题..."):
        # 将用户问题添加到历史记录
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        # --- 标题生成逻辑 ---
        # 如果是首次提问 (历史记录为2条) 且当前对话没有标题

        # 正常继续AI回复流程
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(user_query)
        if len(st.session_state.chat_history) == 2 and st.session_state.active_chat_title is None:
            new_title = generate_chat_title(user_query)
            st.session_state.active_chat_title = new_title
            # 立即保存新标题
            save_chat_history(
                st.session_state.active_chat_title,
                st.session_state.chat_history,
                st.session_state.active_chat_file_path
            )

        # RAG or Web Search
        if st.session_state.rag_active and os.path.exists(INDEX_PATH):
            relevant_files = get_relevant_files(user_query)
            if relevant_files:
                st.session_state.file_selection = {"show": True, "query": user_query, "options": relevant_files}
                st.rerun()

        if not st.session_state.get("file_selection", {}).get("show"):
            with st.chat_message("ai", avatar="💡"):
                with st.spinner("思考中..."):
                    agent_executor = get_agent_executor()
                    response_content = run_agent_with_streaming(agent_executor, user_query,
                                                                st.session_state.chat_history[1:-1])

            st.session_state.chat_history.append(AIMessage(content=response_content))
            # 保存包含AI回复的完整对话
            save_chat_history(
                st.session_state.active_chat_title,
                st.session_state.chat_history,
                st.session_state.active_chat_file_path
            )
            st.rerun()


if __name__ == "__main__":
    main()
