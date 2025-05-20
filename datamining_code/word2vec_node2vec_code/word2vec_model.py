import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import os

# 创建NLTK数据目录（如果不存在）
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# 正确下载必要的NLTK资源
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    print(f"NLTK资源下载错误: {e}")
    print("尝试使用简单的分词方法作为替代...")


# 1. 加载Amazon数据集
def load_amazon_data(file_path):
    # 自定义列名，因为原始CSV没有列名
    column_names = ['category_index', 'review_title', 'review_text']

    # 加载数据，指定没有表头
    df = pd.read_csv(file_path, header=None, names=column_names)

    # 合并标题和评论文本以获得更好的上下文
    df['combined_text'] = df['review_title'].fillna('') + " " + df['review_text'].fillna('')

    return df['combined_text'].tolist()  # 返回合并后的文本列表


# 2. 文本预处理（带有简单分词的备选方案）
def preprocess_text(text_list):
    processed_texts = []

    try:
        # 尝试加载停用词
        stop_words = set(stopwords.words('english'))
    except:
        # 如果加载失败，使用简单的停用词列表
        stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                          "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
                          'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
                          'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
                          'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                          'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
                          'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
                          'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
                          'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                          'with', 'about', 'against', 'between', 'into', 'through', 'during',
                          'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                          'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
                          'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                          'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                          'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                          'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                          "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
                          've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                          "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                          'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
                          'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                          "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
                          'wouldn', "wouldn't"])

    for text in text_list:
        if isinstance(text, str):
            # 转换为小写
            text = text.lower()
            # 去除特殊字符、数字等
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            try:
                # 尝试使用NLTK的分词器
                tokens = word_tokenize(text)
            except:
                # 如果失败，使用简单的空格分词
                tokens = text.split()

            # 去除停用词和单字符词
            filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
            if filtered_tokens:
                processed_texts.append(filtered_tokens)

    return processed_texts


# 3. 训练Word2Vec模型
def train_word2vec(processed_texts, vector_size=100, window=5, min_count=5):
    model = Word2Vec(sentences=processed_texts,
                     vector_size=vector_size,
                     window=window,
                     min_count=min_count,
                     workers=4)
    return model


# 主函数
def main():
    # 指定训练数据集的文件路径
    train_file_path = '/root/datamining/train_part_1.csv'  # 您的训练文件路径

    try:
        # 加载数据
        print("Loading training data...")
        reviews = load_amazon_data(train_file_path)
        print(f"Loaded {len(reviews)} reviews")

        # 预处理文本
        print("Preprocessing text...")
        processed_texts = preprocess_text(reviews)
        print(f"Processed {len(processed_texts)} texts")

        # 训练Word2Vec模型
        print("Training Word2Vec model...")
        model = train_word2vec(processed_texts)
        print(f"Model trained with vocabulary size: {len(model.wv)}")

        # 保存模型
        model_path = "amazon_word2vec.model"
        model.save(model_path)
        print(f"Model saved as '{model_path}'")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()