import os
from dotenv import load_dotenv
load_dotenv()  
import httpx
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

# 加载环境变量
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 页面配置
st.set_page_config(page_title="个人知识库 RAG", page_icon="📚", layout="wide")
st.title("📚 个人知识库 RAG 系统")
st.markdown("上传你的文档（PDF/TXT/Markdown），然后提问，AI 会基于文档内容回答。")

# ------------------------------
# 初始化 Embedding 模型
# ------------------------------
@st.cache_resource
def load_embedding_model():
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./models_cache"
        )
    except Exception as e:
        st.error(f"加载 Embedding 模型失败: {e}")
        st.stop()

# ------------------------------
# 初始化向量库
# ------------------------------
@st.cache_resource
def load_vectorstore(_embedding_model):
    persist_dir = "./chroma_db"
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=_embedding_model)
    else:
        return Chroma(persist_directory=persist_dir, embedding_function=_embedding_model)

# ------------------------------
# 文件 ID 生成（无哈希，可读）
# ------------------------------
def get_file_id(file_path_or_uploaded_file):
    if isinstance(file_path_or_uploaded_file, str):
        abs_path = os.path.abspath(file_path_or_uploaded_file)
        ext = os.path.splitext(abs_path)[1].lower().lstrip('.')
        if "/library/" in abs_path or "\\library\\" in abs_path:
            base = os.path.basename(abs_path)
            name_without_ext = os.path.splitext(base)[0]
            return f"{ext}_library_{name_without_ext}"
        else:
            safe_path = abs_path.replace(os.sep, '_').replace(':', '_')
            return f"{ext}_{safe_path}"
    else:
        name = file_path_or_uploaded_file.name
        ext = os.path.splitext(name)[1].lower().lstrip('.')
        return f"upload_{ext}_{name}"

# ------------------------------
# 从向量库加载所有已处理文件信息（元数据）
# ------------------------------
def load_all_processed_files(vectorstore):
    """返回 { file_id: {"name": ..., "path": ..., "type": ...} }"""
    try:
        all_data = vectorstore.get()
        file_info = {}
        for meta in all_data.get('metadatas', []):
            if meta and 'file_id' in meta:
                fid = meta['file_id']
                if fid not in file_info:
                    file_info[fid] = {
                        "name": meta.get('file_name', '未知文件'),
                        "path": meta.get('file_path', None),
                        "type": meta.get('file_type', 'unknown')
                    }
        return file_info
    except Exception as e:
        print(f"加载已处理文件列表失败: {e}")
        return {}

# ------------------------------
# 删除向量库中指定 file_id 的所有片段
# ------------------------------
def delete_file_from_vectorstore(file_id, vectorstore):
    try:
        vectorstore._collection.delete(where={"file_id": file_id})
        return True
    except Exception as e:
        st.error(f"删除文件时出错: {e}")
        return False

# ------------------------------
# 处理上传的文件（临时文件）
# ------------------------------
def process_uploaded_file(uploaded_file, vectorstore):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        if file_ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif file_ext == ".txt":
            loader = TextLoader(tmp_path, encoding="utf-8")
        elif file_ext in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(tmp_path)
        else:
            st.error(f"不支持的文件类型: {file_ext}")
            os.unlink(tmp_path)
            return 0

        documents = loader.load()
        if not documents:
            st.warning(f"文件 {uploaded_file.name} 未读取到任何内容")
            os.unlink(tmp_path)
            return 0

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        if not chunks:
            chunks = documents

        file_id = get_file_id(uploaded_file)
        file_name = uploaded_file.name

        for chunk in chunks:
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata['file_id'] = file_id
            chunk.metadata['file_name'] = file_name
            chunk.metadata['file_type'] = 'upload'
            chunk.metadata['source'] = uploaded_file.name

        vectorstore.add_documents(chunks)
        os.unlink(tmp_path)

        # 更新 session_state
        if "processed_files_info" in st.session_state:
            st.session_state.processed_files_info[file_id] = {
                "name": file_name,
                "path": None,
                "type": "upload"
            }
        return len(chunks)
    except Exception as e:
        st.error(f"处理上传文件 {uploaded_file.name} 时出错: {e}")
        os.unlink(tmp_path)
        return 0

# ------------------------------
# 处理本地文件（如 library 中的文件）
# ------------------------------
def process_local_file(file_path, vectorstore):
    file_ext = os.path.splitext(file_path)[1].lower()
    try:
        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_ext == ".txt":
            try:
                loader = TextLoader(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                loader = TextLoader(file_path, encoding="gbk")
        elif file_ext in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            return 0

        documents = loader.load()
        if not documents:
            return 0

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        if not chunks:
            chunks = documents

        file_id = get_file_id(file_path)
        file_name = os.path.basename(file_path)
        abs_path = os.path.abspath(file_path)

        for chunk in chunks:
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata['file_id'] = file_id
            chunk.metadata['file_name'] = file_name
            chunk.metadata['file_path'] = abs_path
            chunk.metadata['file_type'] = 'local'
            chunk.metadata['source'] = file_path

        vectorstore.add_documents(chunks)

        # 更新 session_state
        if "processed_files_info" in st.session_state:
            st.session_state.processed_files_info[file_id] = {
                "name": file_name,
                "path": abs_path,
                "type": "local"
            }
        return len(chunks)
    except Exception as e:
        st.error(f"处理本地文件 {os.path.basename(file_path)} 时出错: {e}")
        return 0

# ------------------------------
# 查询改写（规则匹配）
# ------------------------------
def rewrite_query(query: str) -> str:
    q = query.strip().lower()
    rules = {
        "文件内容": "这份文档的主要内容是什么？",
        "内容": "文档讲述了什么内容？",
        "讲了什么": "文档的主要观点和事实有哪些？",
        "总结": "请总结这份文档的核心内容。",
        "摘要": "请为这份文档生成一个简短的摘要。",
        "主题": "这份文档的主题是什么？",
        "主要观点": "文档提出了哪些主要观点或结论？",
    }
    if q in rules:
        return rules[q]
    for fuzzy, semantic in rules.items():
        if fuzzy in q:
            return semantic
    return query

# ------------------------------
# RAG 检索 + 生成回答
# ------------------------------
def rag_answer(retrieval_query, original_query, vectorstore, llm_client, top_k=3):
    docs = vectorstore.similarity_search(retrieval_query, k=top_k)
    if not docs:
        return "知识库中没有找到相关信息，请先上传文档。", []
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    prompt = f"""基于以下资料回答用户的问题。如果资料中没有相关信息，请明确说“根据现有资料无法回答”。
                不要编造事实，尽量引用原文。如果用户询问文档的整体内容，请尝试总结检索到的所有片段。
                对于询问你自身功能或身份的问题（如“你是谁”“你能做什么”），请基于你的常识回答，说明你是知识库助手，功能是基于上传文档回答问题。
                资料：{context}
                问题：{original_query}
                请按以下格式回答：
                    1. [第一个问题的回答]
                    2. [第二个问题的回答]
                    ...
                """
    response = llm_client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "你是一个严谨的知识库助手，基于给定资料回答问题。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    answer = response.choices[0].message.content
    return answer, docs

# ------------------------------
# 主界面
# ------------------------------
def main():
    with st.spinner("初始化 Embedding 模型（首次运行会下载，稍等片刻）..."):
        embed_model = load_embedding_model()
        vectorstore = load_vectorstore(embed_model)

    # 从向量库加载已处理文件信息
    if "processed_files_info" not in st.session_state:
        st.session_state.processed_files_info = load_all_processed_files(vectorstore)

    # 侧边栏
    with st.sidebar:
        st.header("📁 文档管理")

        # ------------------------------
        # 1. 从 library 文件夹选择文件
        # ------------------------------
        library_path = "./library"
        if not os.path.exists(library_path):
            os.makedirs(library_path)
            st.info(f"已创建文件夹 `{library_path}`，请将文档放入其中。")

        supported_exts = (".pdf", ".txt", ".md", ".markdown")

        # 获取所有支持的文件
        all_library_files = []
        if os.path.exists(library_path):
            for f in os.listdir(library_path):
                if f.lower().endswith(supported_exts):
                    all_library_files.append(f)
        all_library_files.sort()

        # 过滤掉已处理过的文件
        library_files = []
        for f in all_library_files:
            file_path = os.path.join(library_path, f)
            file_id = get_file_id(file_path)
            if file_id not in st.session_state.get("processed_files_info", {}):
                library_files.append(f)

        select_options = ["-- 请选择 --"] + library_files + ["📤 上传新文件"]
        selected_option = st.selectbox(
            "选择已有文档，或上传新文件",
            options=select_options,
            key="library_selector"
        )

        # 处理用户的选择（仅处理 library 中的实际文件）
        if selected_option != "-- 请选择 --" and selected_option != "📤 上传新文件":
            file_path = os.path.join(library_path, selected_option)
            file_id = get_file_id(file_path)
            # 这里依然保留检查，防止并发情况（理论上已经过滤，但保留无害）
            if file_id in st.session_state.get("processed_files_info", {}):
                st.info(f"📄 文件 `{selected_option}` 已经在知识库中，无需重复添加。")
            else:
                with st.spinner(f"正在处理 `{selected_option}` ..."):
                    num_chunks = process_local_file(file_path, vectorstore)
                    if num_chunks:
                        st.success(f"✅ 已添加 `{selected_option}`，共 {num_chunks} 个片段")
                    else:
                        st.error(f"❌ 处理 `{selected_option}` 失败")

        # 2. 上传新文件
        if selected_option == "📤 上传新文件":
            uploaded_files = st.file_uploader(
                "上传文档 (PDF/TXT/Markdown)",
                type=["pdf", "txt", "md", "markdown"],
                accept_multiple_files=True,
                key="file_uploader"
            )
            if uploaded_files:
                for file in uploaded_files:
                    file_id = get_file_id(file)
                    if file_id in st.session_state.processed_files_info:
                        st.info(f"📄 文件 `{file.name}` 已经在知识库中，无需重复添加。")
                    else:
                        with st.spinner(f"处理 {file.name} ..."):
                            num_chunks = process_uploaded_file(file, vectorstore)
                            if num_chunks:
                                st.success(f"✅ {file.name} 已添加，共 {num_chunks} 个片段")
                            else:
                                st.error(f"❌ {file.name} 处理失败")

        # 显示知识库片段总数
        try:
            collection_count = vectorstore._collection.count()
            st.info(f"📊 当前知识库片段数: {collection_count}")
        except:
            pass

        # 3. 显示已处理的文件列表（带重新处理按钮）
        st.markdown("---")
        st.subheader("📋 已处理的文件")
        if st.session_state.processed_files_info:
            st.caption(f"共 {len(st.session_state.processed_files_info)} 个文件")
            # 直接显示列表，不使用 expander
            for idx, (fid, info) in enumerate(sorted(st.session_state.processed_files_info.items()), 1):
                col1, col2 = st.columns([4, 1])
                col1.text(f"{idx}. {info['name']}")
                if info['type'] == 'local' and info['path'] and os.path.exists(info['path']):
                    if col2.button("🔄", key=f"reprocess_{fid}"):
                        if delete_file_from_vectorstore(fid, vectorstore):
                            st.session_state.processed_files_info.pop(fid, None)
                            st.success(f"已删除 `{info['name']}` 的旧数据，正在重新处理...")
                            num_chunks = process_local_file(info['path'], vectorstore)
                            if num_chunks:
                                st.success(f"✅ 重新处理 `{info['name']}` 成功，共 {num_chunks} 个片段")
                            else:
                                st.error(f"❌ 重新处理 `{info['name']}` 失败")
                            st.rerun()
                        else:
                            st.error("删除失败")
                else:
                    col2.caption("不可重处理")
        else:
            st.caption("暂无已处理的文件")

        st.markdown("---")
        st.caption("💡 提示：文档会持久化存储在本地 `./chroma_db` 目录中")

    # 聊天界面
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("输入你的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not DEEPSEEK_API_KEY:
            st.error("请先在 .env 文件中设置 DEEPSEEK_API_KEY")
            return

        http_client = httpx.Client()
        llm_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
            http_client=http_client,
        )

        with st.chat_message("assistant"):
            with st.spinner("检索并思考中..."):
                retrieval_query = rewrite_query(prompt)
                answer, source_docs = rag_answer(retrieval_query, prompt, vectorstore, llm_client)
                st.markdown(answer)
                if source_docs:
                    with st.expander("📖 引用来源"):
                        for i, doc in enumerate(source_docs):
                            st.markdown(f"**片段 {i+1}**：\n{doc.page_content[:300]}...")
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()