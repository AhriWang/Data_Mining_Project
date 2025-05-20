from gensim.models import Word2Vec

# 加载你的 Node2Vec 模型
model_path = "/root/datamining/linkedin_node2vec.model"  # 请替换为你的模型路径
node2vec_model = Word2Vec.load(model_path)

# 查看模型训练的前 50 个节点
nodes = list(node2vec_model.wv.index_to_key)
print("Node2Vec 训练的节点（前 50 个）:")
print(nodes[:50])  # 只打印前 50 个
