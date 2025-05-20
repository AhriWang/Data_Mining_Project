import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import gensim

# === 1. 读取 CSV 数据 ===
file_path = "/root/datamining/postings.csv"  # 请修改为你的实际文件路径
df = pd.read_csv(file_path)

# === 2. 选择构造图的列 ===
useful_columns = ["company_name", "title", "location", "formatted_experience_level", "skills_desc", "work_type", "remote_allowed"]
df = df[useful_columns].dropna()  # 去掉缺失值

# === 3. 创建无向图 ===
G = nx.Graph()

# === 4. 添加节点 ===
for col in useful_columns:
    unique_values = df[col].unique()
    G.add_nodes_from(unique_values, type=col)

# === 5. 添加边（构造关系）===
for _, row in df.iterrows():
    G.add_edge(row["company_name"], row["title"])  # 公司 → 职位
    G.add_edge(row["title"], row["location"])  # 职位 → 地点
    G.add_edge(row["title"], row["formatted_experience_level"])  # 职位 → 经验要求
    G.add_edge(row["title"], row["work_type"])  # 职位 → 工作类型
    G.add_edge(row["title"], row["remote_allowed"])  # 职位 → 远程/本地

    # 处理多个技能（假设技能是逗号分隔的）
    skills = str(row["skills_desc"]).split(",")
    for skill in skills:
        skill = skill.strip()
        if skill:
            G.add_edge(row["title"], skill)  # 职位 → 技能

print("\n✅ 图构建完成，节点数:", G.number_of_nodes(), "，边数:", G.number_of_edges())

# === 6. 训练 Node2Vec ===
print("\n🚀 训练 Node2Vec ...")
node2vec = Node2Vec(G, dimensions=128, walk_length=10, num_walks=200, workers=4)
model = node2vec.fit(window=5, min_count=1, batch_words=4)

# === 7. 保存模型 ===
model.save("linkedin_node2vec.model")
print("\n✅ Node2Vec 模型已保存！")


