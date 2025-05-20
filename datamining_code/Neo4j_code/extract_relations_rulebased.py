import pandas as pd
from tqdm import tqdm
from collections import Counter

# 读取实体对数据
df = pd.read_csv("D:\\Datamining\\第五周\\entity_pairs_sample.csv")
relations = []

# 更新后的优化关键词规则
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
                if rel == "treatment":
                    # 如果涉及到治疗，并且实体类型是疾病或症状，则加分
                    if "disease" in type_set or "symptom" in type_set:
                        scores[rel] += 1
                    # 如果是药物与疾病/症状的关系，可以再加权
                    elif "drug" in type_set and ("disease" in type_set or "symptom" in type_set):
                        scores[rel] += 1

                elif rel == "symptom":
                    # 如果是症状关系，且涉及到症状实体，增加额外得分
                    if "symptom" in type_set:
                        scores[rel] += 1
                    # 若症状实体与药物相关联，可以增加得分
                    elif "drug" in type_set and "symptom" in type_set:
                        scores[rel] += 1

                elif rel == "cause":
                    # 如果是疾病和症状之间的因果关系，增加得分
                    if "disease" in type_set and "symptom" in type_set:
                        scores[rel] += 1
                    # 如果疾病和药物之间的因果关系，考虑加分
                    elif "disease" in type_set and "drug" in type_set:
                        scores[rel] += 1

                elif rel == "related_disease":
                    # 如果是相关疾病关系，且涉及到两种疾病，增加得分
                    if "disease" in type_set:
                        scores[rel] += 1

                elif rel == "inhibits":
                    # 如果是抑制关系，检查药物和症状之间的关系
                    if "drug" in type_set and "symptom" in type_set:
                        scores[rel] += 1
                    # 药物与疾病之间的抑制关系，增加得分
                    elif "drug" in type_set and "disease" in type_set:
                        scores[rel] += 1

                elif rel == "side_effect":
                    # 如果是副作用关系，药物与症状之间的副作用关系
                    if "drug" in type_set and "symptom" in type_set:
                        scores[rel] += 1
                    # 副作用与药物、疾病的关系
                    elif "drug" in type_set and "disease" in type_set:
                        scores[rel] += 1

                elif rel == "interacts_with":
                    # 如果是相互作用关系，药物与药物、药物与疾病之间的相互作用关系
                    if "drug" in type_set and ("drug" in type_set or "disease" in type_set):
                        scores[rel] += 1

    return scores.most_common(1)[0][0] if scores else "no_relation"

# 主处理流程
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rule-based Relation Extraction"):
    # 检查 entity1 和 entity2 的长度是否小于8，若是则跳过该行
    if len(str(row['entity1'])) < 8 or len(str(row['entity2'])) < 8:
        continue

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
rel_df.to_csv("entity_relations2.csv", index=False)
print(f"[✓] Rule-based relation extraction completed. Total samples: {len(rel_df)}")
print("[!] Relation types distribution:")
print(rel_df["relation"].value_counts())
