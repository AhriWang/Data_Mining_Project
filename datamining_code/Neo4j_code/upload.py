import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

# 修改为你的实际连接信息
NEO4J_URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "12345678"

# 初始化数据库连接
driver = GraphDatabase.driver(NEO4J_URI, auth=(USERNAME, PASSWORD))

def create_graph(tx, e1, t1, e2, t2, relation):
    # 创建两个节点（带有类型），以及它们之间的关系
    query = f"""
    MERGE (a:{t1} {{name: $e1}})
    MERGE (b:{t2} {{name: $e2}})
    MERGE (a)-[r:{relation.upper()}]->(b)
    """
    tx.run(query, e1=e1, e2=e2)

def import_data(csv_path):
    df = pd.read_csv(csv_path)
    with driver.session() as session:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Importing to Neo4j"):
            e1, t1 = row["entity1"], row["type1"].capitalize()
            e2, t2 = row["entity2"], row["type2"].capitalize()
            rel = row["relation"]
            if rel != "no_relation":
                session.write_transaction(create_graph, e1, t1, e2, t2, rel)

    print("✅ 数据已导入 Neo4j！")

if __name__ == "__main__":
    import_data("D:\Datamining\第五周\cleaned_entity_relations2.csv")
