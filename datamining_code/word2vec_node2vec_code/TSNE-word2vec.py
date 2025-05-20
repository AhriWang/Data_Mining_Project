import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

# 加载 Word2Vec 模型
print("正在加载 Word2Vec 模型...")
model_path = "/root/datamining/word2vecmodel/amazon_word2vec.model"  # 请替换为你的模型路径
model = Word2Vec.load(model_path)
print("Word2Vec 模型加载完成！")

# 获取词向量
words = list(model.wv.index_to_key)[:300]  # 选取前 300 个单词进行降维
word_vectors = np.array([model.wv[word] for word in words])

# 执行 T-SNE 进行降维
print("正在进行 T-SNE 降维...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
word_vec_2d = tsne.fit_transform(word_vectors)
print("T-SNE 计算完成！")

# 绘制散点图
plt.figure(figsize=(14, 10))
plt.scatter(word_vec_2d[:, 0], word_vec_2d[:, 1], alpha=0.5)

# 标注单词
for i, word in enumerate(words):
    plt.annotate(word, (word_vec_2d[i, 0], word_vec_2d[i, 1]), fontsize=9, alpha=0.7)

plt.title("T-SNE Visualization of Word2Vec Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

# 保存图片
save_path = "/root/datamining/tsne_word2vec.png" \
            "" \
            "" \
            ""  # 指定保存路径
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"T-SNE 可视化完成！图像已保存至 {save_path}")
