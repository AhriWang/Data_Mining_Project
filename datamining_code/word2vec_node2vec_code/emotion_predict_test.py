import joblib
from gensim.models import Word2Vec
from emotion_predict import preprocess_text, sentence_vector

# 加载模型
word2vec_model = Word2Vec.load("/root/datamining/word2vecmodel/amazon_word2vec.model")
classifier = joblib.load("/root/datamining/amazon_sentiment_classifier.pkl")


# 预处理 + 向量化
def predict_sentiment(review):
    words = preprocess_text([review])[0]  # 处理文本
    vector = sentence_vector(words, word2vec_model)  # 转向量
    prediction = classifier.predict([vector])[0]

    sentiment_label = "Positive 😊" if prediction == 1 else "Negative 😠"

    # 输出被预测评论和对应的结果
    print(f"\n📢 **评论内容**: \"{review}\"")
    print(f"🔍 **预测结果**: {sentiment_label}")

    return sentiment_label


# 测试
new_review = "This is the worst purchase I’ve ever made. Completely useless and a waste of money."
predict_sentiment(new_review)
