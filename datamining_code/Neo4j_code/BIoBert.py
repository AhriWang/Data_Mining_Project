import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm import tqdm

# === 设置 tqdm 的 pandas 显示（可选）
tqdm.pandas()

# === 步骤 1：加载数据 ===
INPUT_CSV = "cleaned_pubmed_data.csv"
df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=["abstract"]).reset_index(drop=True)

# === 步骤 2：加载预训练的医学 NER 模型（BioBERT）===
model_name = "d4data/biobert-chemical-disease-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# === 步骤 3：构建 transformers pipeline ===
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# === 步骤 4：定义实体提取函数 ===
def extract_entities(text):
    try:
        text = str(text)
        truncated = text[:512]  # BERT 最大长度限制
        results = nlp_ner(truncated)
        entities = [(ent['word'], ent['entity_group']) for ent in results]
        return entities
    except Exception as e:
        return []

# === 步骤 5：提取实体并添加到新列 ===
print("🔍 正在提取实体，请稍等...")
df["entities"] = df.progress_apply(lambda row: extract_entities(f"{row['title']}. {row['abstract']}"), axis=1)

# === 步骤 6：保存结果到新文件 ===
OUTPUT_CSV = "pubmed_ner_result.csv"
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ 提取完成，结果已保存至 {OUTPUT_CSV}，共 {len(df)} 条记录")
