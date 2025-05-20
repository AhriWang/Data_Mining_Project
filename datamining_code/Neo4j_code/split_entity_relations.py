import pandas as pd

# 读取关系数据
df = pd.read_csv("/root/datamining5/entity_relations.csv")

# 只保留有意义的关系（去掉 no_relation）
df = df[df['relation'] != 'no_relation']

# -------------------------
# Step 1: 生成节点文件 nodes.csv
# -------------------------
# 提取实体并去重
entity1_df = df[['entity1', 'type1']].rename(columns={'entity1': 'id', 'type1': 'type'})
entity2_df = df[['entity2', 'type2']].rename(columns={'entity2': 'id', 'type2': 'type'})
nodes_df = pd.concat([entity1_df, entity2_df], ignore_index=True).drop_duplicates()

# 保存节点文件
nodes_df.to_csv("nodes.csv", index=False)
print(f"已生成节点文件 nodes.csv，共 {len(nodes_df)} 个节点")

# -------------------------
# Step 2: 生成关系文件 edges.csv
# -------------------------
edges_df = df[['entity1', 'entity2', 'relation']].rename(columns={
    'entity1': 'source',
    'entity2': 'target'
})

# 保存关系文件
edges_df.to_csv("edges.csv", index=False)
print(f"已生成关系文件 edges.csv，共 {len(edges_df)} 条边")
