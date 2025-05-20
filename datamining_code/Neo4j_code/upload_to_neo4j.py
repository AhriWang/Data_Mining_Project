from neo4j import GraphDatabase
import pandas as pd

# 替换为你的连接信息
NEO4J_URI = "neo4j+s://0a778858.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "tUCeVcRvrVlGVgfungqy9pftlTjCtQA_4y3TFJoT4DQ"

# 加载实体关系数据
df = pd.read_csv("entity_relations.csv")

# 建立驱动
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def upload(tx, row):
    tx.run("""
        MERGE (e1:Entity {name: $entity1, type: $type1})
        MERGE (e2:Entity {name: $entity2, type: $type2})
        MERGE (e1)-[:`$relation`]->(e2)
    """,
    entity1=row['entity1'],
    type1=row['type1'],
    entity2=row['entity2'],
    type2=row['type2'],
    relation=row['relation'])

# 执行上传
with driver.session() as session:
    for _, row in df.iterrows():
        session.execute_write(upload, row)

print("✅ 所有实体关系上传成功！")
