import streamlit as st
from PyPDF2 import PdfReader
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_openai_tools_agent
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun, TavilySearchResults
from langchain.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage
import os
from dotenv import load_dotenv
import shutil
import time
import pickle
from datetime import datetime

from langchain_core.tools import tool

# --- 初始化与配置 ---
load_dotenv(override=True)
st.set_page_config(page_title="RAG 研究助手", page_icon="💡")

# --- 常量定义 ---
DB_DIR = "document_database"
FILES_DIR = os.path.join(DB_DIR, "files")
INDEX_PATH = os.path.join(DB_DIR, "faiss_index")
CHAT_HISTORY_DIR = "chat_history"
ACTIVE_CHAT_FILE = os.path.join(CHAT_HISTORY_DIR, "active_session.pkl")


# --- 后端辅助函数 (无需修改) ---
@st.cache_resource
def get_embeddings():
    # 实际项目中，API Key不应硬编码，而是通过环境变量等方式管理
    return DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))



embeddings = get_embeddings()

def setup_directories():
    os.makedirs(FILES_DIR, exist_ok=True)
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)


def save_chat_history(history, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(history, f)


def load_chat_history(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            pass
    return [AIMessage(content="您好！我是您的AI研究助手。")]

def process_and_index_files(pdf_docs,progress_placeholder):
    """Processes uploaded PDF files, saves them, and indexes their content."""
    texts_with_metadata = []
    progress_bar = progress_placeholder.progress(0, text="开始处理文件...")
    for pdf in pdf_docs:
        # Save the file to the persistent storage
        file_path = os.path.join(FILES_DIR, pdf.name)
        with open(file_path, "wb") as f:
            f.write(pdf.getbuffer())

        # Extract text and create chunks with metadata
        pdf_reader = PdfReader(file_path)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(text)
                for chunk in chunks:
                    # Crucially, add the source filename to each chunk's metadata
                    texts_with_metadata.append((chunk, {"source": pdf.name}))

    if not texts_with_metadata:
        return 0

    # Separate texts and metadatas for FAISS
    texts = [item[0] for item in texts_with_metadata]
    metadatas = [item[1] for item in texts_with_metadata]

    if os.path.exists(INDEX_PATH):
        # Load existing index and add new texts
        db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        db.add_texts(texts, metadatas=metadatas)
    else:
        # Create a new index
        db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    progress_bar.progress(1.0, text="处理完成！")

    db.save_local(INDEX_PATH)
    return len(pdf_docs)


# def process_and_index_files(pdf_docs, progress_placeholder):
#     texts_with_metadata = []
#     progress_bar = progress_placeholder.progress(0, text="开始处理文件...")
#     for i, pdf in enumerate(pdf_docs):
#         progress_bar.progress((i) / len(pdf_docs), text=f"正在处理: {pdf.name}")
#         file_path = os.path.join(FILES_DIR, pdf.name)
#         with open(file_path, "wb") as f:
#             f.write(pdf.getbuffer())
#         try:
#             pdf_reader = PdfReader(file_path)
#             for page in pdf_reader.pages:
#                 text = page.extract_text()
#                 if text:
#                     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#                     chunks = text_splitter.split_text(text)
#                     for chunk in chunks:
#                         texts_with_metadata.append((chunk, {"source": pdf.name}))
#         except Exception as e:
#             st.error(f"处理文件 {pdf.name} 时出错: {e}")
#             continue
#
#     if not texts_with_metadata:
#         progress_bar.empty()
#         st.warning("未能从文件中提取任何文本。")
#         return 0
#
#     progress_bar.progress(0.9, text="正在创建向量索引...")
#     embeddings = get_embeddings()
#     texts = [item[0] for item in texts_with_metadata]
#     metadatas = [item[1] for item in texts_with_metadata]
#     if os.path.exists(INDEX_PATH):
#         db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
#         db.add_texts(texts, metadatas=metadatas)
#     else:
#         db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
#     db.save_local(INDEX_PATH)
#     progress_bar.progress(1.0, text="处理完成！")
#     time.sleep(1)
#     progress_bar.empty()
#     return len(pdf_docs)


def get_relevant_files(query):
    """Searches the entire database to find files relevant to the query."""
    if not os.path.exists(INDEX_PATH):
        return []

    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 20})  # Get top 5 most relevant chunks
    relevant_docs = retriever.get_relevant_documents(query)
    # print(relevant_docs)

    # Extract unique source filenames from metadata
    relevant_files = list(set(doc.metadata['source'] for doc in relevant_docs))
    return relevant_files


def get_agent_executor(selected_files=None):
    """Creates an agent with tools. The document retriever is filtered by selected files."""
    search = TavilySearchResults(max_results=6)
    @tool
    def search_tool(name: str) -> str:
        """当你需要搜索或者查询的时候使用此工具。"""
        print("正在使用工具进行搜索：")
        return search.invoke(name)
    tools = [search_tool]
    # 2. Document Retriever Tool (if files are selected)
    if selected_files:
        full_db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

        # 2. 根据用户的选择，在内存中过滤文档。
        # 我们直接访问 docstore，它保存了所有的文档(Document)对象及其元数据。
        filtered_docs = []
        for doc_id, doc in full_db.docstore._dict.items():
            if doc.metadata.get("source") in selected_files:
                filtered_docs.append(doc)

        # 3. 如果找到了匹配的文档，就用它们在内存中创建一个全新的、临时的 FAISS 索引。
        # 这种方式既高效又能保证检索器只在这些特定的文档中进行搜索。
        if filtered_docs:
            # 从过滤后的文档列表创建新的 FAISS 数据库
            filtered_db = FAISS.from_documents(filtered_docs, embeddings)
            # 基于这个临时的、已过滤的数据库创建检索器
            retriever = filtered_db.as_retriever(search_kwargs={'k': 5})

            # 4. 使用这个被完美过滤的、全新的检索器来创建工具。
            document_retriever = create_retriever_tool(
                retriever,
                "document_retriever",
                f"""请优先使用此工具进行查询和用户问题相关的背景知识。
            这是回答关于文档 '{', '.join(selected_files)}' 具体内容问题的唯一可靠信息来源。
            例如，当被问及“文档中的主要观点是什么？”或“XX项目的截止日期是哪天？”时，应使用此工具。"""
            )
            tools.append(document_retriever)

    llm = init_chat_model("deepseek-chat", model_provider="deepseek")
    # print(tools)

    prompt = hub.pull("hwchase17/openai-tools-agent")

    agent = create_openai_tools_agent(llm, tools, prompt)
    aa = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # print(aa.invoke({"input":"如何评价《愿与愁》"}))
    return aa


# --- 主程序入口 ---
def main():
    setup_directories()

    # 初始化会话状态
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history(ACTIVE_CHAT_FILE)
    if "rag_active" not in st.session_state:
        st.session_state.rag_active = True
    if "file_selection" not in st.session_state:
        st.session_state.file_selection = {"show": False, "query": None, "options": []}
    if "sidebar_view" not in st.session_state:
        st.session_state.sidebar_view = "历史记录"

    # --- 侧边栏 UI ---
    with st.sidebar:
        st.header("RAG 研究助手")
        if st.button("📝 发起新对话", use_container_width=True, type="primary"):
            if os.path.exists(ACTIVE_CHAT_FILE) and len(st.session_state.chat_history) > 1:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_file = os.path.join(CHAT_HISTORY_DIR, f"history_{timestamp}.pkl")
                shutil.copy(ACTIVE_CHAT_FILE, archive_file)
                st.toast(f"先前的对话已存档。", icon="✅")
            st.session_state.chat_history = [AIMessage(content="您好！新的对话已开启。")]
            st.session_state.file_selection = {"show": False, "query": None, "options": []}  # 清空文件选择
            save_chat_history(st.session_state.chat_history, ACTIVE_CHAT_FILE)
            st.rerun()

        st.markdown("---")

        # 侧边栏视图切换
        st.session_state.sidebar_view = st.radio(
            "查看:",
            ["历史记录", "文档数据库"],
            horizontal=True,
            label_visibility="collapsed"
        )

        # --- 动态内容区域 ---
        if st.session_state.sidebar_view == "历史记录":
            st.subheader("🗂️ 历史记录")
            archived_files = sorted([f for f in os.listdir(CHAT_HISTORY_DIR) if f.startswith("history_")], reverse=True)
            if not archived_files:
                st.info("没有历史记录。")
            else:
                for file in archived_files:
                    ts_str = file.replace("history_", "").replace(".pkl", "")
                    label = f"对话于 {datetime.strptime(ts_str, '%Y%m%d_%H%M%S').strftime('%y-%m-%d %H:%M')}"
                    if st.button(label, key=file, use_container_width=True):
                        st.session_state.chat_history = load_chat_history(os.path.join(CHAT_HISTORY_DIR, file))
                        save_chat_history(st.session_state.chat_history, ACTIVE_CHAT_FILE)
                        st.toast(f"已加载会话: {label}")
                        st.rerun()
        else:  # 文档数据库视图
            st.subheader("📚 文档数据库")
            search_term = st.text_input("🔍 搜索文档...", placeholder="输入文件名关键字...",
                                        label_visibility="collapsed")
            indexed_files = [f for f in os.listdir(FILES_DIR) if f.endswith('.pdf')] if os.path.exists(
                FILES_DIR) else []
            if search_term:
                indexed_files = [f for f in indexed_files if search_term.lower() in f.lower()]
            if not indexed_files:
                st.info("数据库为空或未找到匹配项。")
            else:
                st.caption(f"找到 {len(indexed_files)} 个文件")
                for file_name in indexed_files:
                    st.markdown(f"📄 `{file_name}`")

        # --- 固定的文档管理区域 ---
        st.markdown("---")
        with st.expander("⚙️ 文档管理"):
            pdf_docs = st.file_uploader("上传新PDF", accept_multiple_files=True, type=['pdf'])
            progress_placeholder = st.empty()
            if st.button("处理并索引", disabled=not pdf_docs, use_container_width=True):
                count = process_and_index_files(pdf_docs, progress_placeholder)
                if count > 0:
                    st.success(f"成功处理并索引了 {count} 个新文件。")

            st.markdown("---")
            if st.button("🔥 清空所有数据", use_container_width=True):
                confirm_check = st.checkbox("我确认要永久删除所有文档和历史记录。")
                if confirm_check:
                    if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)
                    if os.path.exists(CHAT_HISTORY_DIR): shutil.rmtree(CHAT_HISTORY_DIR)
                    st.session_state.chat_history = [AIMessage(content="所有数据均已清除。")]
                    st.success("所有数据已被清空。")
                    time.sleep(1)
                    st.rerun()

    # --- 主聊天界面 (始终显示) ---
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.subheader("当前对话")
    with col2:
        st.session_state.rag_active = st.toggle("激活 RAG", value=st.session_state.rag_active,
                                                help="激活后，将优先在您上传的文档中寻找答案。")

    # RAG 文件选择界面
    if st.session_state.file_selection.get("show"):
        with st.chat_message("AI", avatar="💡"):
            st.info("根据您的问题，我找到了这些可能相关的文档。")
            selected_files = st.multiselect(
                "请选择您希望我使用的文件：",
                options=st.session_state.file_selection["options"],
                default=st.session_state.file_selection["options"]
            )
            if st.button("基于所选文件生成回答", type="primary"):
                with st.spinner("思考中..."):
                    query = st.session_state.file_selection["query"]
                    st.session_state.chat_history.append(HumanMessage(content=query))
                    agent_executor = get_agent_executor(selected_files)
                    response = agent_executor.invoke(
                        {"input": "请优先检索本地文件后回答下面的问题:"+query, "chat_history": st.session_state.chat_history[1:-1]})
                    st.session_state.chat_history.append(AIMessage(content=response['output']))
                    save_chat_history(st.session_state.chat_history, ACTIVE_CHAT_FILE)
                    st.session_state.file_selection["show"] = False
                    st.rerun()
    else:
        # 聊天记录展示
        for message in st.session_state.chat_history:
            avatar = "💡" if isinstance(message, AIMessage) else "🧑‍💻"
            with st.chat_message(message.type, avatar=avatar):
                st.write(message.content)

    # 用户输入处理
    if user_query := st.chat_input("请输入您的问题..."):
        if st.session_state.rag_active and os.path.exists(INDEX_PATH):
            relevant_files = get_relevant_files(user_query)
            if relevant_files:
                st.session_state.file_selection = {"show": True, "query": user_query, "options": relevant_files}
                st.rerun()
            else:
                st.toast("未在您的文档库中找到相关信息，将尝试联网搜索。", icon="🌐")

        if not st.session_state.file_selection.get("show"):
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            with st.chat_message("user", avatar="🧑‍💻"):
                st.write(user_query)
            with st.spinner("思考中..."):
                agent_executor = get_agent_executor()
                # print(st.session_state.chat_history[1:-1])
                response = agent_executor.invoke(
                    {"input": user_query, "chat_history": st.session_state.chat_history[1:-1]})
                st.session_state.chat_history.append(AIMessage(content=response['output']))
                save_chat_history(st.session_state.chat_history, ACTIVE_CHAT_FILE)
                st.rerun()


if __name__ == "__main__":
    main()
