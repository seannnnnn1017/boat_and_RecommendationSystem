from sentence_transformers import SentenceTransformer, util

# 載入預訓練的 SBERT 模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 要比較的兩個句子
sentence1 = "AI課程"
sentence2 = "機器學習導論"
sentence3 = "世界經濟論壇"
# 將句子編碼為語意向量
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)
embedding3 = model.encode(sentence3, convert_to_tensor=True)
# 計算餘弦相似度
cosine_score = util.pytorch_cos_sim(embedding1, embedding2)

print(f"AI課程 和 機器學習導論 的相似度為: {cosine_score.item():.4f}")
cosine_score1 = util.pytorch_cos_sim(embedding1, embedding3)

print(f"AI課程 和 世界經濟論壇 的相似度為: {cosine_score1.item():.4f}")