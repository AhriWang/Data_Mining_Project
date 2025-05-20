import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import nltk
nltk.download('punkt')

# 1️⃣ 加载 Word2Vec 词向量模型
word2vec_model = Word2Vec.load("/root/datamining/word2vecmodel/amazon_word2vec.model")


# 2️⃣  加载 Amazon 数据集（无表头，手动加列名）
def load_amazon_data(file_path):
    column_names = ["label", "review_title", "review_text"]
    df = pd.read_csv(file_path, header=None, names=column_names)

    # 合并标题和评论文本
    df["combined_text"] = df["review_title"].fillna("") + " " + df["review_text"].fillna("")

    # 统一 label（0: 负面, 1: 正面）
    df["label"] = df["label"].map({1: 0, 2: 1})

    return df["combined_text"].tolist(), df["label"].values


# 3️⃣ 文本预处理（去除特殊字符、分词、去停用词）
nltk.download("punkt")
nltk.download("stopwords")


def preprocess_text(text_list):
    stop_words = set(stopwords.words("english"))
    processed_texts = []

    for text in text_list:
        if isinstance(text, str):
            text = text.lower()  # 转小写
            text = re.sub(r"[^a-zA-Z\s]", "", text)  # 去除特殊字符
            words = word_tokenize(text)  # 分词
            words = [w for w in words if w not in stop_words and len(w) > 1]  # 去除停用词
            processed_texts.append(words)

    return processed_texts


# 4️⃣ 将评论转换为固定大小的向量
def sentence_vector(words, model):
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


# 5️⃣ 训练情感分类器（随机森林）
def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("🎯 分类模型准确率:", accuracy_score(y_test, y_pred))

    return clf


# 6️⃣ 主函数
def main():
    train_file_path = "/root/datamining/train_part_1.csv"

    print("🔹 加载数据...")
    reviews, labels = load_amazon_data(train_file_path)
    print(f"✅ 加载 {len(reviews)} 条评论")

    print("🔹 文本预处理...")
    processed_texts = preprocess_text(reviews)
    print(f"✅ 处理 {len(processed_texts)} 条评论")

    print("🔹 计算评论向量...")
    X_vectors = np.array([sentence_vector(words, word2vec_model) for words in processed_texts])

    print("🔹 训练情感分类模型...")
    classifier = train_classifier(X_vectors, labels)
    print("✅ 训练完成！")

    # 保存分类模型
    joblib.dump(classifier, "amazon_sentiment_classifier.pkl")
    print("✅ 分类器已保存")


if __name__ == "__main__":
    main()
