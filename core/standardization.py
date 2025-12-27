import json
import os

import joblib
import numpy as np

from core.lyric_analyzer import LyricsProcessor


class Standardizer:
    """数据标准化类"""

    def __init__(self, scaler_path: str):
        """
        :param scaler_path: 标准化器保存路径，用于后续新数据
        """
        self.scaler_path = scaler_path
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

    def init_standardizer(self, features_matrix: np.ndarray):
        """
        初始化阶段：基于足够的训练数据计算标准化参数
        参数:
            features_matrix: numpy数组，形状为 (n_samples, n_features)
        返回:
            标准化后的特征矩阵
        """
        # 1. 初始化标准化器
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()

        # 2. 计算并应用标准化
        # fit_transform = 计算每个特征的均值和标准差，然后进行转换
        features_scaled = self.scaler.fit_transform(features_matrix)

        # 3. 验证标准化效果
        print("-" * 60 + " 标准化验证 " + "-" * 60)
        print(f"原始数据 - 各特征均值: {np.mean(features_matrix, axis=0)[:5]}")  # 显示前5个特征
        print(f"原始数据 - 各特征标准差: {np.std(features_matrix, axis=0)[:5]}")
        print(f"标准化后 - 各特征均值: {np.mean(features_scaled, axis=0)[:5]}")
        print(f"标准化后 - 各特征标准差: {np.std(features_scaled, axis=0)[:5]}")
        return features_scaled

    def save(self):
        """保存标准化器"""
        with open(self.scaler_path, 'wb') as f:
            joblib.dump(self.scaler, f)

    def process_new_features(self, new_song_features: np.ndarray):
        """
        处理新特征的特征标准化
        参数:
            new_song_features: 新特征的原始特征向量 (1, n_features)
        返回:
            标准化后的特征向量
        """
        if not self.scaler:
            raise ValueError("请先初始化标准化器 or 加载已有标准化器")
        # 应用相同的标准化变换
        new_features_scaled = self.scaler.transform(new_song_features.reshape(1, -1))
        return new_features_scaled.flatten()

    def batch_process_new_features(self, new_songs_features: np.ndarray):
        """
        批量处理新特征的特征标准化
        参数:
            new_songs_features: 新特征的批量特征向量 (n_songs, n_features)
        返回:
            标准化后的特征向量
        """
        # 应用相同的标准化变换
        new_features_scaled = self.scaler.transform(new_songs_features)
        return new_features_scaled

    @staticmethod
    def reduce_dimension(embeddings, target_dim):
        """将向量维度降低到目标维度"""
        # 方法A：PCA降维
        from sklearn.decomposition import PCA
        pca = PCA(n_components=target_dim, random_state=42)
        return pca.fit_transform(embeddings)


if __name__ == "__main__":
    max_features = 86

    # 获取歌词特征
    with open('../测试数据/歌词关键词列表.json', 'r', encoding='utf-8') as f:
        lyrics_keywords_list = json.load(f)

    print("\n" + "=" * 60 + " 歌词关键词特征批量获取 " + "=" * 60)
    processor = LyricsProcessor()
    lyrics_embeddings = processor.bert_encode(lyrics_keywords_list)
    print(f"歌词特征维度: {lyrics_embeddings.shape}")

    print("\n" + "=" * 60 + " 降维 " + "=" * 60)
    lyrics_reduced = processor.reduce_lyrics_dimension(lyrics_embeddings, target_dim=max_features)
    print(f"降维后的歌词特征维度: {lyrics_reduced.shape}")

    # 特征标准化
    print("\n" + "=" * 60 + " 特征标准化 " + "=" * 60)
    features_matrix = lyrics_reduced
    scaler_path = "../测试数据/standardizer.pkl"
    standardizer = Standardizer(scaler_path)
    features_scaled = standardizer.init_standardizer(np.array(features_matrix))
    print(f"标准化后的特征维度: {features_scaled.shape}")
    standardizer.save()
