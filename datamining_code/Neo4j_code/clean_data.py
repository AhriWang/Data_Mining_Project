import pandas as pd
import re

# === 读取你之前生成的文件 ===
df = pd.read_csv("D:\\Datamining\\第五周\\pubmed_titles_abstracts.csv")

# === 文本清洗函数 ===
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"\s+", " ", text)                      # 去除多余空格、换行
    text = re.sub(r"[^a-zA-Z0-9.,;:()\[\]\-\' ]", "", text)  # 移除特殊字符（保留常见英文标点）
    text = text.lower().strip()                           # 转小写 + 去前后空格
    return text

# === 应用于 title 和 abstract 两列 ===
df['title'] = df['title'].apply(clean_text)
df['abstract'] = df['abstract'].apply(clean_text)

# === 过滤掉摘要太短或缺失的记录（可选） ===
df = df[df['abstract'].str.len() > 50].reset_index(drop=True)

# === 保存为新文件 ===
df.to_csv("cleaned_pubmed_data.csv", index=False)

print(f"✅ 清洗完成，共 {len(df)} 条文献已保存至 cleaned_pubmed_data.csv")
