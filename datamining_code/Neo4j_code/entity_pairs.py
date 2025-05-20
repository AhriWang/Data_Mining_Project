import pandas as pd
from itertools import combinations
from tqdm import tqdm

# 加载NER结果
df = pd.read_csv("/root/datamining5/pubmed_ner_results.csv")  # 替换为你的实际文件名
pairs = []

# 遍历每一行，带进度条
for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取实体对"):
    abstract = row['abstract']
    entities_raw = row['entities']

    if not isinstance(entities_raw, str):
        continue  # 跳过空或不是字符串的行

    entity_list = entities_raw.split("; ")
    entities = []

    for item in entity_list:
        if "(" in item and ")" in item:
            try:
                ent = item[:item.rfind("(")].strip()
                label = item[item.rfind("(")+1:item.rfind(")")]
                entities.append((ent, label))
            except:
                continue

    for e1, e2 in combinations(entities, 2):
        if e1 != e2:
            pairs.append({
                'title': row['title'],
                'abstract': abstract,
                'entity1': e1[0],
                'type1': e1[1],
                'entity2': e2[0],
                'type2': e2[1]
            })

# 保存前5000条为 CSV
pairs_df = pd.DataFrame(pairs)
pairs_df.head(5000).to_csv("entity_pairs_sample.csv", index=False)
print("实体对提取完成，已保存前5000条，共提取：", len(pairs_df), "条")
