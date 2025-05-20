import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

# 1️⃣ 加载 Node2Vec 训练好的模型
model_path = "/root/datamining/linkedin_node2vec.model"  # 你的模型路径
model = Word2Vec.load(model_path)

# 2️⃣ 提取所有节点 ID 及其对应的向量
node_ids = list(model.wv.index_to_key)  # 获取所有节点 ID
node_vectors = np.array([model.wv[node] for node in node_ids])  # 提取所有节点的嵌入向量

# 3️⃣ 进行 KMeans 聚类分析
num_clusters = 5  # 设定聚类数量
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(node_vectors)

# 4️⃣ 使用 t-SNE 进行降维（2D 可视化）
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
node_vectors_2d = tsne.fit_transform(node_vectors)

# 5️⃣ 绘制 t-SNE 可视化并标注聚类类别
plt.figure(figsize=(10, 8))
scatter = plt.scatter(node_vectors_2d[:, 0], node_vectors_2d[:, 1], c=clusters, cmap="tab10", alpha=0.7)

# 添加图例
plt.legend(handles=scatter.legend_elements()[0], labels=[f"Cluster {i}" for i in range(num_clusters)])
plt.title("t-SNE Visualization of Node2Vec Embeddings with KMeans Clustering")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

# 6️⃣ 保存图片到指定路径
save_path = "/root/datamining/node2vec_kmeans_tsne.png"
plt.savefig(save_path, dpi=300)
plt.close()

print(f"✅ t-SNE 可视化已保存至 {save_path}")
