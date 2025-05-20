import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from multiprocessing import Pool, cpu_count
import time
import os

# 下载NLTK资源
nltk.download('punkt')
nltk.download('stopwords')

# 全局变量
STOP_WORDS = set(stopwords.words('english'))


# 文本预处理函数 - 用于并行处理
def process_text(text):
    if not isinstance(text, str):
        return None

    # 转小写
    text = text.lower()
    # 去除特殊字符
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 分词
    tokens = word_tokenize(text)
    # 过滤停用词和单字符词
    filtered_tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]

    return filtered_tokens if filtered_tokens else None


# 批量加载和处理数据
def process_data_in_batches(file_path, batch_size=10000):
    # 自定义列名
    column_names = ['category_index', 'review_title', 'review_text']

    # 计算总行数
    total_rows = sum(1 for _ in open(file_path))
    print(f"Total rows in dataset: {total_rows}")

    # 创建一个ChunkReader
    chunks = pd.read_csv(file_path,
                         header=None,
                         names=column_names,
                         chunksize=batch_size)

    all_processed_texts = []
    processed_count = 0
    start_time = time.time()

    # 利用多进程加速处理
    num_cores = max(1, cpu_count() - 1)  # 留一个核心给系统
    pool = Pool(processes=num_cores)

    print(f"Processing data using {num_cores} CPU cores...")

    for i, chunk in enumerate(chunks):
        # 合并标题和评论文本
        chunk['combined_text'] = chunk['review_title'].fillna('') + " " + chunk['review_text'].fillna('')
        texts = chunk['combined_text'].tolist()

        # 并行处理这个批次
        processed_batch = pool.map(process_text, texts)

        # 过滤掉None值
        processed_batch = [text for text in processed_batch if text is not None]
        all_processed_texts.extend(processed_batch)

        # 更新进度
        processed_count += len(chunk)
        elapsed_time = time.time() - start_time
        progress = processed_count / total_rows * 100

        print(f"Processed batch {i + 1}, {processed_count}/{total_rows} "
              f"({progress:.2f}%), Time elapsed: {elapsed_time:.2f}s")

    pool.close()
    pool.join()

    return all_processed_texts


# 训练Word2Vec模型
def train_word2vec(processed_texts, vector_size=100, window=5, min_count=5):
    print("Training Word2Vec model...")
    model = Word2Vec(sentences=processed_texts,
                     vector_size=vector_size,
                     window=window,
                     min_count=min_count,
                     workers=cpu_count() - 1)
    return model


# 主函数
def main():
    # 训练数据集文件路径
    train_file_path = '/root/datamining/train_part_1.csv'

    # 检查文件是否存在
    if not os.path.exists(train_file_path):
        print(f"Error: File '{train_file_path}' not found.")
        return

    # 优化内存使用的批处理
    processed_texts = process_data_in_batches(train_file_path)
    print(f"Total processed texts: {len(processed_texts)}")

    # 训练模型
    model = train_word2vec(processed_texts)
    print(f"Model trained with vocabulary size: {len(model.wv)}")

    # 保存模型
    model.save("amazon_word2vec.model")
    print("Model saved as 'amazon_word2vec.model'")


if __name__ == "__main__":
    main()