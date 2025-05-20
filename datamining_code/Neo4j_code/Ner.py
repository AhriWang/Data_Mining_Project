import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm import tqdm

# 本地模型路径
model_path = "D:\\Datamining\\第五周\\biomedical-ner-all"

# 加载本地模型和分词0器0
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# 初始化 NER pipeline'0[11[[
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# 输入文件（包含 title 和 abstract 列）
input_csv = "D:\\Datamining\\第五周\\cleaned_pubmed_data.csv"
# 输出文件
output_csv = "pubmed_ner_results.csv"

# 读取数据
df = pd.read_csv(input_csv)
df.fillna("", inplace=True)  # 避免空值报错

# 新列存储识别结果
df["entities"] = ""

# 批量抽取实体
for idx, row in tqdm(df.iterrows(), total=len(df), desc="NER识别中"):
    text = row["title"] + " " + row["abstract"]
    entities = ner_pipeline(text)
    # 格式化为字符串保存
    formatted_entities = [f"{e['word']}({e['entity_group']})" for e in entities]
    df.at[idx, "entities"] = "; ".join(formatted_entities)

# 保存结果
df.to_csv(output_csv, index=False)
print(f"命名实体识别完成，已保存到 {output_csv}")
