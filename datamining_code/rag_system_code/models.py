import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os


@st.cache_resource
def load_embedding_model(model_name):
    """Loads the sentence transformer model."""
    st.write(f"Loading embedding model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        st.success("Embedding model loaded.")
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None


@st.cache_resource
def load_generation_model(model_name):
    """Loads the Hugging Face generative model and tokenizer."""
    st.write(f"Loading generation model: {model_name}...")
    try:
        # 设置环境变量以解决accelerate问题
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # 先加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 修改模型加载参数，避免device_map和low_cpu_mem_usage问题
        has_gpu = torch.cuda.is_available()

        if has_gpu:
            # 如果有GPU，使用这些参数
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,  # 使用半精度
                device_map="auto"  # 让transformers自动处理
            )
        else:
            # 如果只有CPU，禁用low_cpu_mem_usage，不使用device_map
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=False,  # 禁用这个选项避免报错
                torch_dtype=torch.float32  # CPU上使用float32
            )

        st.success("Generation model and tokenizer loaded.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load generation model: {e}")
        import traceback
        st.error(traceback.format_exc())  # 添加完整错误堆栈以便调试
        return None, None