import json

import numpy as np


class LyricsProcessor:
    """歌词处理器"""
    _instance = None
    _bert_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        import os
        # 设置镜像源（必须在导入其他库之前）可选
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        # 使用轻量级BERT模型获取歌词语义向量
        from sentence_transformers import SentenceTransformer
        self._bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def bert_encode(self, lyrics_list: str | list[str] | np.ndarray):
        """将关键词列表转化为文本并编码"""
        return self._bert_model.encode(lyrics_list)

    def reduce_lyrics_dimension(self, lyrics_embeddings, target_dim):
        """将384维的语义向量降到与音频向量相近的维度"""
        # 方法A：PCA降维
        from sklearn.decomposition import PCA
        pca = PCA(n_components=target_dim, random_state=42)
        lyrics_reduced = pca.fit_transform(lyrics_embeddings)

        # 方法B：UMAP降维（保持局部结构）
        # reducer = umap.UMAP(n_components=target_dim, random_state=42)
        # lyrics_reduced = reducer.fit_transform(lyrics_embeddings)

        return lyrics_reduced


if __name__ == "__main__":
    max_features = 86

    # 歌词关键词可通过 AI 获取
    with open('../测试数据/歌词关键词列表.json', 'r', encoding='utf-8') as f:
        lyrics_keywords_list = json.load(f)

    print("\n" + "=" * 60 + " 歌词关键词特征批量获取 " + "=" * 60)
    processor = LyricsProcessor()
    lyrics_embeddings = processor.bert_encode(lyrics_keywords_list)
    print(f"歌词特征维度: {lyrics_embeddings.shape}")

    print("\n" + "=" * 60 + " 降维 " + "=" * 60)
    lyrics_reduced = processor.reduce_lyrics_dimension(lyrics_embeddings, target_dim=max_features)
    print(f"降维后的歌词特征维度: {lyrics_reduced.shape}")
