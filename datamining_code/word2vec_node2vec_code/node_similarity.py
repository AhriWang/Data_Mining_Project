from gensim.models import Word2Vec

# 加载 Node2Vec 模型
node2vec_model = Word2Vec.load("/root/datamining/linkedin_node2vec.model")

# 查看所有节点（前10个）
print("Node2Vec 训练的节点（前30个）:", node2vec_model.wv.index_to_key[:30])

# 计算两个节点的相似度
node1, node2 =  "Senior Software Support Analyst", "Oracle Financial Consultant"  # 请替换为你的节点ID
similarity = node2vec_model.wv.similarity(node1, node2)
print(f"Node2Vec: 节点 '{node1}' 与 '{node2}' 的相似度: {similarity}")

node3, node4 =  "Lead Data Engineer", ".Net Core Developer"  # 请替换为你的节点ID
similarity = node2vec_model.wv.similarity(node3, node4)
print(f"Node2Vec: 节点 '{node3}' 与 '{node4}' 的相似度: {similarity}")

# 找到与某个节点最相似的5个节点
target_node1 = "Project Manager"  # 请替换为你的节点ID
similar_nodes = node2vec_model.wv.most_similar(target_node1, topn=5)
print(f"与节点 '{target_node1}' 最相似的5个节点:", similar_nodes)

target_node2 = "United States"  # 请替换为你的节点ID
similar_nodes = node2vec_model.wv.most_similar(target_node2, topn=5)
print(f"与节点 '{target_node2}' 最相似的5个节点:", similar_nodes)
