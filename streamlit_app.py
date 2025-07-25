from io import StringIO
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_tool_calling_agent
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

from langchain_experimental.tools import PythonAstREPLTool

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
    return DashScopeEmbeddings(model="text-embedding-v3", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))


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

def process_uploaded_files(uploaded_files):
    pdf_docs = [f for f in uploaded_files if f.name.lower().endswith('.pdf')]
    csv_files = [f for f in uploaded_files if f.name.lower().endswith('.csv')]

    pdf_count, csv_loaded_name = 0, None
    progress_placeholder = st.empty()

    if pdf_docs:
        with st.spinner(f"正在处理 {len(pdf_docs)} 个PDF文件..."):
            pdf_count = process_and_index_files(pdf_docs, progress_placeholder)

    if csv_files:
        csv_to_load = csv_files[0]
        try:
            st.session_state.df = pd.read_csv(csv_to_load)
            st.session_state.data_analysis_messages = [
                AIMessage(content=f"已成功加载 `{csv_to_load.name}`。现在您可以开始对这个表格提问了。")]
            st.session_state.dff = True
            if os.path.exists('plot.png'): os.remove('plot.png')
            csv_loaded_name = csv_to_load.name
            if len(csv_files) > 1:
                st.toast(f"加载了第一个CSV: {csv_loaded_name} 用于分析。", icon="⚠️")

        except Exception as e:
            st.error(f"加载CSV文件 '{csv_to_load.name}' 失败: {e}")

    progress_placeholder.empty()
    summary = []
    if pdf_count > 0: summary.append(f"成功处理了 {pdf_count} 个PDF。")
    if csv_loaded_name:
        summary.append(f"加载了 '{csv_loaded_name}' 用于数据分析。")


    if summary:
        st.success(" ".join(summary)); time.sleep(2)
    else:
        st.warning("没有上传有效的文件类型（PDF或CSV）。")


def get_relevant_files(query):
    """根据查询在整个数据库中检索相关文件名"""
    if not os.path.exists(INDEX_PATH):
        return []

    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 20})
    relevant_docs = retriever.get_relevant_documents(query)

    return list(set(doc.metadata['source'] for doc in relevant_docs))


def get_agent_executor(selected_files=None,df=None):
    """创建带有工具的 Agent Executor。如果选择了文件，则文档检索器会被过滤"""
    search = TavilySearchResults(max_results=3)

    @tool
    def search_tool(query: str) -> str:
        """当你需要搜索最新信息或用户问题与本地文档无关时使用此工具。"""
        return search.invoke(query)

    tool1 = search_tool
    locals_dict = {'df': st.session_state.df}
    tool2 = PythonAstREPLTool(locals=locals_dict)


    tools = [tool1]
    if not st.session_state.data_active:
        pass
    else:
        tools.append(tool2)

    if df:
        aa = None
    else:
        aa = st.session_state.df.head().to_markdown()

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
            tool3 = document_retriever_tool
            tools.append(tool3)


    llm = init_chat_model("deepseek-chat", model_provider="deepseek", temperature=0.7)

    system_prompt = f"""
## 角色与任务

你是一个功能强大的多功能智能体。你的主要目标是分析用户的请求，从你的工具库中选择最合适的工具，执行它，然后根据工具的输出和自己的思考形成最终答案。

## 工具库

有以下三种工具,但你可以使用里面的若干种工具,根据实际情况决定：

1.  **`web_search`**

      * **功能**: 在互联网上搜索实时或常识性信息。
      * **何时使用**: 当问题涉及当前事件、事实或任何未明确包含在用户私有文件中的主题时使用。这是你处理一般性查询的默认工具。

2.  **`document_retriever`**

      * **功能**: 从用户已上传的PDF文档中检索特定信息。
      * **何时使用**: 仅当用户的问题是关于他们上传的文件内容时使用（例如，“在我的文档中”、“根据这篇论文”、“总结这个文件”）。

3.  **`python_data_analyzer`**

      * **功能**: 在一个已加载的CSV文件上执行Python代码，该文件可作为一个名为 `df` 的Pandas DataFrame 使用。
      * **何时使用**: 仅当问题需要基于已加载的数据集进行计算、数据分析或绘图时使用（例如，“计算平均值”、“绘制图表”、“分析数据”）。
      * **[当前情境]**
          * 一个CSV文件当前已被加载。DataFrame `df` 可用于分析。
          * 以下是 `df.head().to_markdown()` 的输出，供你参考,如果输出为空，代表此时没有'df'供你访问;相反，你可以访问完整的 `df`。
    ```
    {aa}
    ```
``* **[绘图指南]** * 如果用户要求绘图 
1.  **使用 `matplotlib.pyplot` (别名为 `plt`)** 来创建图表。
2.  **为每张图保存为唯一文件**: 每生成一张图，都必须将其保存为一个唯一的 `.png` 文件。例如 `pic\\age_histogram.png`, `pic\\salary_plot.png`。不要重复使用 `pic\\plot.png`。
3.  **最终输出格式**: 当你完成所有绘图并保存文件后，你的最终、唯一的输出必须遵循下面的特殊格式。对于你生成的每一张图，都创建一个`GRAPH_BEGIN/GRAPH_END`块。

    **格式模板:**
    ```text
    GRAPH_BEGIN
    file: [你保存的第一个文件名.png]
    title: [第一张图的中文标题]
    GRAPH_END
    GRAPH_BEGIN
    file: [你保存的第二个文件名.png]
    title: [第二张图的中文标题]
    GRAPH_END
    ```
    **示例**: 用户问: "画出年龄的直方图和薪水的箱线图"
    你的代码工具调用会执行两次保存: `plt.savefig('age_hist.png')` 和 `plt.savefig('salary_box.png')`。
    然后，你的最终输出**必须是**:
    ```text
    GRAPH_BEGIN
    file: age_hist.png
    title: 用户年龄分布直方图
    GRAPH_END
    GRAPH_BEGIN
    file: salary_box.png
    title: 用户薪水箱线图
    GRAPH_END
    ```
4.  **注意**: 绘制图表时请使用英文，因为服务器环境可能缺少中文字体。但在 `title:` 部分请使用中文。

## 关键输出指令
**重要提示：** 你的所有面向用户的最终回答、解释和总结都**必须**使用**简体中文**。你的内部思考过程可以是英文，但交付给用户的最终答案必须是流畅自然的中文。

现在，请回答用户的问题。
"""
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}"), ("placeholder", "{agent_scratchpad}")])

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
    """重置会话状态以开始一个新的临时对话，但先不保存文件。"""
    st.session_state.active_chat_file_path = None  # 关键改动：路径为None表示未保存
    st.session_state.active_chat_title = "新对话" # 可以给一个临时标题
    st.session_state.chat_history = [AIMessage(content="您好！新的对话已开启，有什么可以帮您？")]
    st.session_state.file_selection = {"show": False, "query": None, "options": []}
    st.rerun()


@st.dialog("数据详情")
def show_data_dialog():
    """用于显示当前DataFrame信息的弹窗"""
    st.markdown('<div class="custom-wide-dialog"></div>', unsafe_allow_html=True)
    st.markdown("#### 数据预览 (前10行)")
    st.dataframe(st.session_state.df.head(10))
    st.markdown("#### 数据维度")
    st.write(f"{st.session_state.df.shape[0]} 行 × {st.session_state.df.shape[1]} 列")
    st.markdown("#### 数据列信息")
    buffer = StringIO()
    st.session_state.df.info(buf=buffer)
    st.text(buffer.getvalue())
    if st.button("关闭", use_container_width=True):
        st.rerun()
    # st.markdown('</div>', unsafe_allow_html=True)

def render_message_content(content: str):
    """
    解析消息内容并以合适的方式在 Streamlit 中渲染。
    如果内容包含 GRAPH_BEGIN 标记，则将其解析为文本和图表。
    否则，直接显示为 Markdown 文本。
    """
    # 检查内容中是否包含图表标记
    if "GRAPH_BEGIN" in content:
        # 分离可能的文本部分和图表定义部分
        parts = content.split("GRAPH_BEGIN", 1)
        text_analysis_part = parts[0].strip()
        graph_definition_part = "GRAPH_BEGIN" + parts[1]

        # 1. 如果有文本部分，先渲染文本
        if text_analysis_part:
            st.markdown(text_analysis_part)

        # 2. 解析并渲染图表
        try:
            graphs_data = graph_definition_part.strip().split("GRAPH_BEGIN")
            for graph_block in graphs_data:
                if "GRAPH_END" in graph_block:
                    file_path = None
                    title = "未命名图表"
                    lines = graph_block.strip().split('\n')
                    for line in lines:
                        if line.startswith("file:"):
                            file_path = line.replace("file:", "").strip()
                        if line.startswith("title:"):
                            title = line.replace("title:", "").strip()

                    if file_path and os.path.exists(file_path):
                        st.markdown(f"**{title}**")
                        st.image(file_path)
                    elif file_path:
                        st.error(f"图表渲染失败：找不到文件 {file_path}")
        except Exception as e:
            st.error(f"解析图表数据时出错: {e}")
            st.text(content) # 如果解析失败，显示原始文本以便调试
    else:
        # 如果没有图表标记，直接将内容作为 Markdown 显示
        st.markdown(content)

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
    if 'df' not in st.session_state:
        st.session_state.df = None
        st.session_state.dff = False

    # --- 侧边栏 UI ---
    with st.sidebar:
        # st.divider()


        st.title("RAG Agent")
        # 当对话历史>1条时（即用户已提问），才允许新建对话
        # a=st.empty()
        is_chatting = len(st.session_state.get("chat_history", [])) > 1
        if st.button("📝 新建对话", use_container_width=True, disabled=not is_chatting,type="primary"):
            start_new_chat()
        # st.divider()
        # st.subheader("🗓️ 对话历史")
        all_chats = sorted(
            [f for f in os.listdir(CHAT_HISTORY_DIR) if f.startswith("chat_") and f.endswith(".pkl")],
            reverse=True
        )
        st.divider()
        # st.subheader("🗓️ 对话历史")
        st.button("🗓️ 近期对话", use_container_width=True)

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

        uploaded_files = st.file_uploader("上传文件", accept_multiple_files=True,label_visibility="collapsed")

        if st.button("▶️ 开始处理", disabled=not uploaded_files, use_container_width=True,type="primary"):
            csv_loaded_name = process_uploaded_files(uploaded_files)
            # print(st.session_state.df)
            st.rerun()
        # print(ccc)
        if st.button("👀 查看数据详情",disabled=not st.session_state.dff, use_container_width=True,type="primary"):
            show_data_dialog()
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
            if st.button("🗑️ 清空所有数据", use_container_width=True, type="primary"):

                if 'confirm_delete' not in st.session_state:
                    st.session_state.confirm_delete = False

                @st.dialog("确认删除")
                def confirm_delete_dialog():
                    st.warning("⚠️ 您确定要删除所有数据吗？此操作无法撤销！")
                    col1, col2 = st.columns([0.5,0.5])
                    with col1:
                        if st.button("取消",type="primary", use_container_width=True):
                            st.session_state.confirm_delete = False
                            st.rerun()

                    with col2:
                        if st.button("确认删除", type="primary", use_container_width=True):
                            st.session_state.confirm_delete = True
                            if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)
                            if os.path.exists(CHAT_HISTORY_DIR): shutil.rmtree(CHAT_HISTORY_DIR)
                            st.session_state.clear()
                            st.success("数据已清空。")
                            time.sleep(1)
                            st.rerun()
                confirm_delete_dialog()


        st.divider()

    # --- 主聊天界面 ---
    for message in st.session_state.get("chat_history", []):
        avatar = "💡" if isinstance(message, AIMessage) else "🧑‍💻"
        with st.chat_message(message.type, avatar=avatar):
            render_message_content(message.content)
            # st.write(message.content)

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
                        agent_executor = get_agent_executor(selected_files=selected_files,df= not st.session_state.dff)
                        final_query = f"请先判断是否需要提供的 `document_retriever` 工具来回答以下问题，如果需要，请先调用`document_retriever`进行背景知识的检索。问题是：{query}"
                        response_content = run_agent_with_streaming(agent_executor, final_query,
                                                                    st.session_state.chat_history[1:-1])

                st.session_state.chat_history.append(AIMessage(content=response_content))
                # 实时保存到当前激活的对话文件中
                save_chat_history(st.session_state.active_chat_title, st.session_state.chat_history,
                                  st.session_state.active_chat_file_path)
                st.session_state.file_selection["show"] = False
                st.rerun()
    with st._bottom:
        controls_container = st.container()
        with controls_container:
            # 使用列布局将控件组放置在左侧
            view_cols = st.columns([0.6, 0.4])
            with view_cols[0]:
                st.markdown("**AI 模式选择:**")
                # 在左侧列内部再次使用列，让两个开关紧挨着
                control_cols = st.columns(2)
                with control_cols[0]:
                    st.session_state.rag_active = st.toggle("文档问答", value=False, help="启用此模式后，AI会优先从您上传的PDF文档中寻找答案。")
                with control_cols[1]:
                    # is_df_loaded = st.session_state.get('dff', False)
                    st.session_state.data_active = st.toggle("数据分析", value=False, disabled=not is_df_loaded, help="需先上传CSV文件。启用后，AI可以对表格数据进行计算和绘图。")
        # coo,coo1,jiy = st.columns([0.2,0.2,0.6])
        # with coo:
        #     st.session_state.rag_active = st.toggle("RAG",value=False)
        # with coo1:
        #     st.session_state.data_active = st.toggle("Data", value=False)


    if user_query := st.chat_input("请输入您的问题..."):

        is_new_chat = st.session_state.active_chat_file_path is None

        # 如果是新对话，在处理之前先“固化”它
        if is_new_chat:
            # 1. 为新对话创建真实的文件路径和标题
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_chat_filename = f"chat_{timestamp}.pkl"
            st.session_state.active_chat_file_path = os.path.join(CHAT_HISTORY_DIR, new_chat_filename)

            st.session_state.active_chat_title = generate_chat_title(user_query)

            # 2. 将包含第一条用户消息的会话历史首次写入文件
            # 注意：此时 chat_history 里已经有AI的欢迎语了，现在追加第一条用户消息
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            save_chat_history(
                st.session_state.active_chat_title,
                st.session_state.chat_history,
                st.session_state.active_chat_file_path
            )
        else:
            # 对于已存在的对话，只需追加用户消息
            st.session_state.chat_history.append(HumanMessage(content=user_query))

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
                    agent_executor = get_agent_executor(df= not st.session_state.dff)
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
