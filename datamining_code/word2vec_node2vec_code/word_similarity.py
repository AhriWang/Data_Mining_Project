from gensim.models import Word2Vec

# 加载 Word2Vec 模型
word2vec_model = Word2Vec.load("/root/datamining/word2vecmodel/amazon_word2vec.model")

# 查看词表中的前10个单词
print("Word2Vec 词表中的前10个单词:", word2vec_model.wv.index_to_key[:10])

# 计算两个单词的相似度
word1, word2 = "good", "excellent"
similarity = word2vec_model.wv.similarity(word1, word2)
print(f"Word2Vec: '{word1}' 与 '{word2}' 的相似度: {similarity}")

word3, word4 = "awful", "bad"
similarity = word2vec_model.wv.similarity(word3, word4)
print(f"Word2Vec: '{word3}' 与 '{word4}' 的相似度: {similarity}")

# 找到与某个单词最相似的5个单词
target_word1 = "cola"
similar_words = word2vec_model.wv.most_similar(target_word1, topn=5)
print(f"与 '{target_word1}' 最相似的5个词:", similar_words)

target_word2 = "apple"
similar_words = word2vec_model.wv.most_similar(target_word2, topn=5)
print(f"与 '{target_word2}' 最相似的5个词:", similar_words)
