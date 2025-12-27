"""
创建蓝图
"""
import os

from flask import Blueprint, request

from server.controller.result import Result
from server.service.feature_extraction_service import FeatureExtractionService

feature_extraction_bp = Blueprint("feature-extraction", __name__, url_prefix="/feature-extraction")

"""
特征提取服务
"""
feature_extraction_service = FeatureExtractionService()


@feature_extraction_bp.post('/audio-feature-extraction')
def audio_feature_extraction():
    """
    音频特征提取接口
    请求参数：
    - filepath: 音频文件路径
    返回结果：
    - 包含各类音频特征的字典
    """
    audio_filepath: str = request.json.get('audioFilepath')

    if not os.path.exists(audio_filepath):
        return Result.fail(message="音频文件不存在")

    mfcc_features, spectral_features, temporal_features, energy_features, music_features = feature_extraction_service.audio_feature_extraction(
        audio_filepath)
    data = {
        "mfccFeatures": mfcc_features,
        "spectralFeatures": spectral_features,
        "temporalFeatures": temporal_features,
        "energyFeatures": energy_features,
        "musicFeatures": music_features,
    }
    return Result.success(result=data)


@feature_extraction_bp.post('/lyrics-feature-extraction')
def lyrics_feature_extraction():
    """
    歌词特征提取接口
    请求参数：
    - model_filepath: 模型文件路径
    - lyrics_list: 歌词列表
    返回结果：
    - 提取的歌词特征
    """
    body = request.json
    lyrics_list: list = body.get('lyricsList')

    if lyrics_list is None or len(lyrics_list) <= 0:
        return Result.fail(message="缺少必要参数")

    data = feature_extraction_service.lyrics_feature_extraction(
        lyrics_list=lyrics_list
    )
    return Result.success(result=data)


@feature_extraction_bp.post('/init-feature-standardizer')
def init_feature_standardizer():
    """
    初始化特征标准化器接口
    请求参数：
    - scaler_filepath: 标准化器保存路径
    - features_matrix_filepath: 特征矩阵文件路径
    - features_scaled_filepath: 标准化后特征矩阵保存路径
    返回结果：
    - 成功消息
    """
    body = request.json
    scaler_filepath: str = body.get('scalerFilepath')
    features_matrix_filepath: str = body.get('featuresMatrixFilepath')
    features_scaled_filepath: str = body.get('featuresScaledFilepath')
    max_features: int = body.get('maxFeatures')

    if not all([scaler_filepath, features_matrix_filepath, features_scaled_filepath]):
        return Result.fail(message="缺少必要参数")
    if not scaler_filepath:
        return Result.fail(message="模型文件参数不可为空")
    if not os.path.exists(features_matrix_filepath):
        return Result.fail(message="特征矩阵文件不存在")
    if not features_scaled_filepath:
        return Result.fail(message="特征矩阵结果文件参数不可为空")

    feature_extraction_service.init_feature_standardizer(
        scaler_filepath=scaler_filepath,
        features_matrix_filepath=features_matrix_filepath,
        features_scaled_filepath=features_scaled_filepath,
        max_features=max_features
    )
    return Result.success()


@feature_extraction_bp.post('/feature-standardization')
def feature_standardization():
    """
    特征标准化接口
    请求参数：
    - scaler_filepath: 标准化器文件路径
    - features_matrix: 特征矩阵
    返回结果：
    - 标准化后的特征矩阵
    """
    body = request.json
    scaler_filepath: str = body.get('scalerFilepath')
    features_matrix: list = body.get('featuresMatrix')

    if not scaler_filepath:
        return Result.fail('请提供有效的标准化器文件路径')
    if features_matrix is None:
        return Result.fail('请提供有效的特征矩阵')
    if not os.path.exists(scaler_filepath):
        return Result.fail(message="标准化器模型文件不存在")

    data = feature_extraction_service.feature_standardization(
        scaler_filepath=scaler_filepath,
        features_matrix=features_matrix
    )
    return Result.success(result=data)
