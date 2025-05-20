import streamlit as st
import time
import os
from rag_core_rerank import generate_answer, rerank

# 设置 HuggingFace 镜像和缓存路径
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './hf_cache'

# 引入配置和工具函数
from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, TOP_K,
    MAX_ARTICLES_TO_INDEX, MILVUS_LITE_DATA_PATH, COLLECTION_NAME,
    id_to_doc_map
)
from data_utils import load_data
from models import load_embedding_model, load_generation_model
from milvus_utils import get_milvus_client, setup_milvus_collection, index_data_if_needed, search_similar_documents
from rag_core import generate_answer

# --- 初始化页面配置 ---
st.set_page_config(layout="wide")
st.title("📄 医疗 RAG 多轮对话系统 (Milvus Lite)")
st.markdown(f"使用 Milvus Lite, `{EMBEDDING_MODEL_NAME}`, 和 `{GENERATION_MODEL_NAME}`")

# 初始化对话历史
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 获取 Milvus 客户端
milvus_client = get_milvus_client()

if milvus_client:
    collection_is_ready = setup_milvus_collection(milvus_client)
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    generation_model, tokenizer = load_generation_model(GENERATION_MODEL_NAME)

    models_loaded = embedding_model and generation_model and tokenizer

    if collection_is_ready and models_loaded:
        pubmed_data = load_data(DATA_FILE)
        if pubmed_data:
            indexing_successful = index_data_if_needed(milvus_client, pubmed_data, embedding_model)
        else:
            st.warning(f"无法从 {DATA_FILE} 加载数据，跳过索引。")
            indexing_successful = False

        st.divider()

        if not indexing_successful and not id_to_doc_map:
            st.error("数据索引失败或不完整，且没有文档映射。RAG 功能已禁用。")
        else:
            st.subheader("💬 多轮问答交互")
            query = st.text_input("请输入您的提问（支持多轮对话）:", key="query_input")

            if st.button("获取答案", key="submit_button") and query:
                start_time = time.time()

                with st.spinner("正在搜索相关文档..."):
                    retrieved_ids, distances = search_similar_documents(milvus_client, query, embedding_model)

                if not retrieved_ids:
                    st.warning("找不到相关文档。")
                else:
                    retrieved_docs = [id_to_doc_map[id] for id in retrieved_ids if id in id_to_doc_map]

                    if not retrieved_docs:
                        st.error("检索结果无映射文档。")
                    else:
                        # 先rerank，注意这里是rerank函数，返回已排序的文档列表
                        retrieved_docs = rerank(query, retrieved_docs, top_k=TOP_K)

                        st.subheader("📄 经过 rerank 排序后的上下文文档")
                        for i, doc in enumerate(retrieved_docs):
                            # 这里距离和ID你要自己调整，因为rerank没返回id和距离，如果需要显示可改rerank返回值结构
                            with st.expander(f"文档 {i + 1} - {doc['title'][:60]}"):
                                st.write(f"**标题:** {doc['title']}")
                                st.write(f"**摘要:** {doc['abstract']}")

                        st.divider()

                        # 拼接对话历史
                        dialogue_context = ""
                        for past_q, past_a in st.session_state.chat_history:
                            dialogue_context += f"用户：{past_q}\nAI：{past_a}\n"
                        full_prompt = dialogue_context + f"用户：{query}\nAI："

                        st.subheader("🧠 生成的答案")
                        with st.spinner("生成答案中..."):
                            answer = generate_answer(full_prompt, retrieved_docs, generation_model, tokenizer)
                            st.write(answer)

                        # 添加到对话历史
                        st.session_state.chat_history.append((query, answer))

                end_time = time.time()
                st.info(f"耗时: {end_time - start_time:.2f} 秒")

            # 显示历史对话
            # 显示历史对话（分角色气泡）
            if st.session_state.chat_history:
                st.divider()
                st.subheader("📜 对话历史")

                for i, (q, a) in enumerate(st.session_state.chat_history):
                    with st.container():
                        st.markdown(
                            f"""
                            <div style='background-color:#e6f0ff;padding:10px;border-radius:10px;margin-bottom:10px'>
                                <b>🧑 用户：</b><br>{q}
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown(
                            f"""
                            <div style='background-color:#f9f9f9;padding:10px;border-left:4px solid #409EFF;border-radius:10px;margin-bottom:20px'>
                                <b>🤖 AI：</b><br>{a}
                            </div>
                            """, unsafe_allow_html=True
                        )

                if st.button("🗑️ 清空对话历史"):
                    st.session_state.chat_history = []
                    st.experimental_rerun()


    else:
        st.error("模型加载或 collection 设置失败。请检查配置。")
else:
    st.error("Milvus 客户端初始化失败。")

# --- 侧边栏 ---
st.sidebar.header("系统配置")
st.sidebar.markdown(f"**向量存储:** Milvus Lite")
st.sidebar.markdown(f"**数据路径:** `{MILVUS_LITE_DATA_PATH}`")
st.sidebar.markdown(f"**Collection:** `{COLLECTION_NAME}`")
st.sidebar.markdown(f"**数据文件:** `{DATA_FILE}`")
st.sidebar.markdown(f"**嵌入模型:** `{EMBEDDING_MODEL_NAME}`")
st.sidebar.markdown(f"**生成模型:** `{GENERATION_MODEL_NAME}`")
st.sidebar.markdown(f"**最大索引数:** `{MAX_ARTICLES_TO_INDEX}`")
st.sidebar.markdown(f"**检索 Top K:** `{TOP_K}`")
