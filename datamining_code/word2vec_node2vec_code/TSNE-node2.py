import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 1️⃣ 读取 job_embeddings.csv
file_path = "/root/datamining/job_embeddings.csv"  # 请修改为你的实际路径
df = pd.read_csv(file_path)

# 2️⃣ 提取 job_id（转换为字符串）和对应的向量
job_ids = df.iloc[:, 0].astype(str).values  # 确保 job_id 是字符串
embeddings = df.iloc[:, 1:].values  # 其余列是向量

# 3️⃣ 选择部分数据进行绘制（防止图像过密）
sample_size = 500  # 只取 500 个点，避免过于密集
if len(job_ids) > sample_size:
    indices = np.random.choice(len(job_ids), sample_size, replace=False)  # 随机采样
    job_ids = job_ids[indices]
    embeddings = embeddings[indices]

# 4️⃣ 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# 5️⃣ 绘制散点图
plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10, alpha=0.7, c="blue")

# 6️⃣ 仅标注部分点（最多 30 个）
num_labels = min(30, len(job_ids))  # 限制最多 30 个标注
label_indices = np.random.choice(len(job_ids), num_labels, replace=False)  # 随机选取要标注的点

for i in label_indices:
    plt.annotate(job_ids[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8, alpha=0.75)

# 7️⃣ 设置图表信息
plt.title("t-SNE Visualization of Job Embeddings (Sampled)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

# 8️⃣ 保存图片
save_path = "/root/datamining/job_tsne_sampled.png"
plt.savefig(save_path, dpi=300)
plt.close()

print(f"✅ t-SNE 可视化已保存至 {save_path}")
