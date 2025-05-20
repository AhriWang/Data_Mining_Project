import pandas as pd
from tqdm import tqdm
from collections import Counter

# 读取实体对数据
df = pd.read_csv("/root/datamining5/entity_pairs_sample.csv")
relations = []

# 优化后的关键词规则
rules = {
    "treatment": [
        "treat", "used for", "therapy", "relieve", "management",
        "administer", "intervention", "prescribed", "effective against", "ameliorate"
    ],
    "symptom": [
        "symptom", "experience", "feel", "pain", "suffer",
        "manifest", "complain of", "present with", "report"
    ],
    "cause": [
        "cause", "lead to", "induce", "result in", "trigger",
        "provoke", "precipitate", "contribute to"
    ],
    "related_disease": [
        "associated with", "linked to", "related to", "co-occur",
        "overlap with", "risk factor for"
    ],
    "inhibits": [
        "inhibits", "blocks", "suppresses", "prevents", "counteracts"
    ],
    "side_effect": [
        "adverse effect", "leads to nausea", "headache", "side effect", "toxicity", "allergic reaction"
    ],
    "interacts_with": [
        "interacts with", "co-administered with", "drug interaction", "combination with"
    ]
}

# 根据实体类型决定一些打分条件
def get_type_set(t1, t2):
    return {t1.lower(), t2.lower()}

# 匹配打分逻辑
def match_relation(abstract, t1, t2):
    scores = Counter()
    type_set = get_type_set(t1, t2)

    for rel, keywords in rules.items():
        for kw in keywords:
            if kw in abstract:
                # 基础得分
                scores[rel] += 1

                # 某些关系考虑实体类型的加权
                if rel == "treatment" and ("disease" in type_set or "symptom" in type_set):
                    scores[rel] += 1
                elif rel == "symptom" and "symptom" in type_set:
                    scores[rel] += 1
                elif rel == "related_disease" and "disease" in type_set:
                    scores[rel] += 1

    return scores.most_common(1)[0][0] if scores else "no_relation"

# 主处理流程
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rule-based Relation Extraction"):
    abstract = str(row['abstract']).lower()
    relation = match_relation(abstract, row['type1'], row['type2'])

    relations.append({
        "title": row["title"],
        "abstract": row["abstract"],
        "entity1": row["entity1"],
        "type1": row["type1"],
        "entity2": row["entity2"],
        "type2": row["type2"],
        "relation": relation
    })

# 保存结果
rel_df = pd.DataFrame(relations)
rel_df.to_csv("entity_relations.csv", index=False)
print(f"[✓] Rule-based relation extraction completed. Total samples: {len(rel_df)}")
print("[!] Relation types distribution:")
print(rel_df["relation"].value_counts())
