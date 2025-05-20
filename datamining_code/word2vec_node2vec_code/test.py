import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import os
import swifter  # ✅ 用于并行加速 apply()

# === 下载并配置 NLTK 停用词 ===
nltk.data.path.append('/root/nltk_data')
nltk.download('stopwords', force=True)

# ✅ 将停用词设置为全局变量，减少重复加载
STOP_WORDS = set(stopwords.words('english'))

# === 读取 CSV 文件并设置列名 ===
train_df1 = pd.read_csv('/root/datamining/train_part_1.csv', names=['label', 'review_title', 'review_text'], header=None)
train_df2 = pd.read_csv('/root/datamining/train_part_2.csv', names=['label', 'review_title', 'review_text'], header=None)
test_df = pd.read_csv('/root/datamining/test.csv', names=['label', 'review_title', 'review_text'], header=None)

# === 合并数据集 ===
df = pd.concat([train_df1, train_df2, test_df], ignore_index=True)

# === 合并标题和文本列 ===
df['text'] = df['review_title'].astype(str) + ' ' + df['review_text'].astype(str)

# ✅ 定义预处理函数（用正则表达式替代 word_tokenize）
def preprocess_text(text):
    # 将文本转换为小写
    text = text.lower()
    # 去除标点符号和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 分词（用正则表达式替代 word_tokenize）
    tokens = re.findall(r'\w+', text)
    # 去除停用词
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return tokens

# ✅ 使用 swifter 进行并行加速 apply()
print("\n🚀 正在预处理文本数据（使用 swifter 加速）...")
df['processed_text'] = df['text'].swifter.apply(preprocess_text)

# === 使用 TfidfVectorizer 进行分词 ===
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,    # 直接使用分词后的结果
    lowercase=False,          # 已经小写化
    token_pattern=None        # 忽略默认的正则匹配
)
X = vectorizer.fit_transform(df['processed_text'])

# === 将分词结果转换为列表，供 Word2Vec 使用 ===
sentences = df['processed_text'].tolist()

# ✅ 构建 Word2Vec 模型（利用多线程加速）
print("\n🚀 正在训练 Word2Vec 模型...")
model = Word2Vec(
    sentences=sentences,      # 使用分词后的结果
    vector_size=100,          # 词向量维度
    window=5,                 # 上下文窗口大小
    min_count=2,              # 最低出现次数
    workers=os.cpu_count(),   # 使用 CPU 最大线程数
    epochs=20                 # 训练轮数
)

# ✅ 训练模型（可以适当增大 epochs 以提升效果）
model.train(sentences, total_examples=len(sentences), epochs=20)

# === 保存模型 ===
model.save('/root/datamining/amazon_word2vec.model')
print("\n✅ Word2Vec 模型已保存！")

# === 加载模型 ===
model = Word2Vec.load('/root/datamining/amazon_word2vec.model')
print("\n✅ Word2Vec 模型已加载！")

# === 示例1：输出某个词的词向量 ===
word = 'good'
if word in model.wv:
    print(f"\nVector for '{word}':")
    print(model.wv[word])
else:
    print(f"\n'{word}' not in vocabulary")

# === 示例2：查看与某个词最相似的词 ===
if word in model.wv:
    print(f"\nMost similar words to '{word}':")
    print(model.wv.most_similar(word))

# === 示例3：获取词汇表大小 ===
print("\nVocabulary size:")
print(len(model.wv.index_to_key))
