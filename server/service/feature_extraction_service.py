import json
import os

import numpy as np

from core.audio_features import AudioFeatureExtractor
from core.lyric_analyzer import LyricsProcessor
from core.standardization import Standardizer


class FeatureExtractionService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        pass

    def audio_feature_extraction(self, audio_filepath):
        """音频特征提取"""
        print("=" * 60 + " 开始提取音频特征... " + "=" * 60)
        extractor = AudioFeatureExtractor(audio_filepath)

        # MFCC 特征，80维
        print("-" * 60 + " 1. 提取 MFCC 特征 " + "-" * 60)
        mfcc_features = extractor.extract_mfcc_features()
        print(f"MFCC 特征示例 (前10个): {mfcc_features[:10]}")

        # 频谱特征，共 6 维
        print("-" * 60 + " 2. 提取频谱特征 " + "-" * 60)
        spectral_features = extractor.extract_spectral_features()
        print(f"频谱特征示例: {spectral_features}")

        # 时域特征 3维
        print("-" * 60 + " 3. 提取时域特征 " + "-" * 60)
        temporal_features = extractor.extract_temporal_features()
        print(f"时间特征示例: {temporal_features}")

        # 能量特征，4维
        print("-" * 60 + " 4. 提取能量特征 " + "-" * 60)
        energy_features = extractor.extract_energy_features()
        print(f"能量特征示例: {energy_features}")

        # 音乐特征，14维
        print("-" * 60 + " 5. 提取音乐特征 " + "-" * 60)
        music_features = extractor.extract_music_features()
        print(f"音乐特征示例 (前10个): {music_features[:10]}")
        return mfcc_features, spectral_features, temporal_features, energy_features, music_features

    def lyrics_feature_extraction(self, lyrics_list: list[str]):
        """歌词特征提取"""
        # 创建歌词特征处理器
        _processor = LyricsProcessor()
        # 歌词特征提取
        return _processor.bert_encode(lyrics_list)

    def reduce_lyrics_dimension(self, lyrics_embeddings_filepath: str, lyrics_reduced_filepath: str, target_dim):
        """将384维的语义向量降到与音频向量相近的维度"""
        with open(lyrics_embeddings_filepath, 'r', encoding='utf-8') as f:
            lyrics_embeddings = np.array(json.load(f))
        # 创建歌词特征处理器
        _processor = LyricsProcessor()
        lyrics_reduced_list = _processor.reduce_lyrics_dimension(lyrics_embeddings, target_dim=target_dim)
        with open(lyrics_reduced_filepath, 'w', encoding='utf-8') as fp:
            json.dump(lyrics_reduced_list, fp)

    def init_feature_standardizer(self, scaler_filepath: str, features_matrix_filepath: str,
                                  features_scaled_filepath: str, max_features: int):
        """初始化特征标准化器"""
        print("-" * 60 + " 特征标准化器初始化 start " + "-" * 60)
        if os.path.exists(scaler_filepath):
            os.remove(scaler_filepath)
            print(f"已删除已存在的特征标准化器文件: {scaler_filepath}")

        # 读取特征矩阵文件
        with open(features_matrix_filepath, 'r', encoding='utf-8') as f:
            features_matrix = np.array(json.load(f)).astype(np.float64)
        # 初始化特征标准化器
        standardizer = Standardizer(scaler_path=scaler_filepath)
        features_scaled = standardizer.init_standardizer(features_matrix)

        # 保存特征标准化器
        standardizer.save()

        # 降维
        if max_features and features_scaled.shape[1] > max_features:
            print(f'{"-" * 60 + " 降维 start " + "-" * 60}')
            print(f"特征矩阵维度: {features_scaled.shape}")
            features_scaled = standardizer.reduce_dimension(features_scaled, target_dim=max_features)
            print(f"降维后的特征矩阵维度: {features_scaled.shape}")
            print(f'{"-" * 60 + " 降维 end " + "-" * 60}')

        if os.path.exists(features_scaled_filepath):
            os.remove(features_scaled_filepath)
            print(f"已删除已存在的特征矩阵文件: {features_scaled_filepath}")

        # 保存特征矩阵
        with open(features_scaled_filepath, 'w', encoding='utf-8') as fp:
            json.dump(features_scaled.tolist(), fp)
        print("-" * 60 + " 特征标准化器初始化 end " + "-" * 60)

    def feature_standardization(self, scaler_filepath: str, features_matrix: list):
        """特征标准化"""
        if not os.path.exists(scaler_filepath):
            raise FileNotFoundError(f"模型文件不存在: {scaler_filepath}")
        standardizer = Standardizer(scaler_path=scaler_filepath)
        return standardizer.batch_process_new_song(np.array(features_matrix))
