import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

# 1️⃣ 加载已训练的 node2vec .model 文件
model_path = "/root/datamining/linkedin_node2vec.model"  # 替换为你的模型路径
model = Word2Vec.load(model_path)

# 2️⃣ 提取所有节点 ID 及其对应的向量
node_ids = list(model.wv.index_to_key)  # 获取所有节点 ID
node_vectors = np.array([model.wv[node] for node in node_ids])  # 提取所有节点的嵌入向量

# 3️⃣ 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, perplexity=30, random_state=42)  # perplexity 影响点的分布
node_vectors_2d = tsne.fit_transform(node_vectors)

# 4️⃣ 绘制 t-SNE 可视化
plt.figure(figsize=(10, 8))
plt.scatter(node_vectors_2d[:, 0], node_vectors_2d[:, 1], s=10, alpha=0.7, c="blue")

# 选取部分节点进行标注
num_labels = 20  # 仅标注部分节点
for i in range(num_labels):
    plt.annotate(node_ids[i], (node_vectors_2d[i, 0], node_vectors_2d[i, 1]), fontsize=8, alpha=0.75)

plt.title("Node2Vec Embeddings Visualization using t-SNE")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

# 5️⃣ 保存图片到指定路径
save_path = "/root/datamining/node2vec_tsne.png"
plt.savefig(save_path, dpi=300)
plt.close()

print(f"✅ t-SNE 可视化已保存至 {save_path}")