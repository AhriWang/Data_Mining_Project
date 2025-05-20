import streamlit as st
import torch
from config import MAX_NEW_TOKENS_GEN, TEMPERATURE, TOP_P, REPETITION_PENALTY
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource(show_spinner=False)
def load_reranker():
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
    return tokenizer, model

def rerank(query, documents, top_k=5):
    """
    重新排序文档列表 documents，按与 query 的相关性降序排列
    documents: List[dict]，每个包含 'content' 字段
    """
    rerank_tokenizer, rerank_model = load_reranker()
    texts = [doc['content'] for doc in documents]
    inputs = rerank_tokenizer(
        [query] * len(texts), texts,
        padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        scores = rerank_model(**inputs).logits.view(-1)
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranked[:top_k]]

def generate_answer(query, context_docs, gen_model, tokenizer):
    """Generates an answer using the LLM based on query and context."""
    if not context_docs:
        return "I couldn't find relevant documents to answer your question."
    if not gen_model or not tokenizer:
         st.error("Generation model or tokenizer not available.")
         return "Error: Generation components not loaded."

    # 对文档进行 rerank 选择最相关的 top_k
    context_docs = rerank(query, context_docs, top_k=3)

    # 合并上下文内容
    context = "\n\n---\n\n".join([doc['content'] for doc in context_docs])

    prompt = f"""Based ONLY on the following context documents, answer the user's question.
If the answer is not found in the context, state that clearly. Do not make up information.

Context Documents:
{context}

User Question: {query}

Answer:
"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(gen_model.device)
        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_GEN,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        st.error(f"Error during text generation: {e}")
        return "Sorry, I encountered an error while generating the answer."
