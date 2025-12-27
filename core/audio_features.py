import json
import os
from typing import Optional

import librosa
import librosa.display
import numpy as np

from utils.json_codec import CustomEncoder


class AudioFeatureExtractor:
    """音频特征提取类"""

    def __init__(self, audio_path: str,
                 sr: float = None,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mfcc: int = 20,
                 n_mels: int = 256,
                 delta: bool = True,
                 delta2: bool = True,
                 n_bands: int = 6,
                 fmin: float = 200.0,
                 ):
        """
        提取音频文件的MFCC特征

        参数:
        ----------
        audio_path : str
            音频文件路径
        sr : float
            目标采样率，None表示使用原始采样率 (默认: None)
            - 高音质音乐保持44.1kHz
            - 语音内容降采样到16kHz
            - 统一重采样到22.05kHz，平衡音乐和语音需求，计算效率较高
        n_fft : int
            FFT窗口大小，单位为样本数 (默认: 2048)，决定频率分辨率和时间分辨率的权衡
            - 音乐分析标准：2048（44.1kHz采样率时对应46ms）
            - 节奏感强的音乐：1024-2048（平衡时间和频率分辨率）
            - 古典/舒缓音乐：4096（更精细的和声分析）
            - 实时推荐：1024（更快计算）
        hop_length : int
            帧移大小，单位为样本数 (默认: 512)，决定时间分辨率和特征冗余度
            - 标准设置：512（当n_fft=2048时）
            - 高时间精度：256（适合鼓点分析）
            - 计算优化：1024（适合长音乐推荐）
            - 重叠率公式：hop_length = n_fft / 4（推荐）
        n_mfcc : int
            要提取的MFCC系数数量
            - n_mfcc=13: 语音为主的推荐（播客、有声书）
            - n_mfcc=20: 音乐为主的推荐（歌曲推荐）
            - n_mfcc=26: 混合内容推荐
        n_mels : int
            梅尔滤波器的数量 (默认: 256),决定频谱的压缩程度和特征维度
            - 标准设置：128（平衡细节和计算）
            - 音色敏感推荐：256（如乐器识别推荐）
            - 语音为主推荐：80（节省计算资源）
            - 环境音乐推荐：64-80（关注主要频段）
        delta : bool
            是否计算一阶差分 (Delta MFCC) (默认: True)，物理意义：捕捉动态特征
            - 一阶差分，表征MFCC随时间的变化率
            - delta=True：几乎所有音乐推荐都应启用
        delta2 : bool
            是否计算二阶差分 (Delta-Delta MFCC) (默认: False)，物理意义：捕捉动态特征
            - 二阶差分，表征变化率的变化（加速度）
        n_bands : int

        fmin : float

        """
        self.audio_path = audio_path
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.delta = delta
        self.delta2 = delta2

        # 添加文件存在性检查
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        # 忽略警告
        import warnings
        warnings.filterwarnings("ignore", message="PySoundFile failed.*")
        warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.*")

        try:
            y, original_sr = librosa.load(audio_path, sr=sr, mono=True)
            print(f"音频加载成功: {audio_path}， original_sr: {original_sr}")
            self.y = y
            self.sr = sr if sr else original_sr
            print(f"采样率: {self.sr} Hz, 音频长度: {len(self.y) / self.sr:.2f} 秒, 样本数: {len(self.y)}")
            # 根据采样率调整频带数量
            self.n_bands = n_bands  # 默认值
            self.fmin = fmin
            if self.sr <= 8000:
                self.n_bands = 4
                self.fmin = 100.0
            elif self.sr <= 16000:
                self.n_bands = 5
                self.fmin = 200.0
        except Exception as e:
            raise RuntimeError(
                f"无法加载音频文件 {audio_path}，请检查文件格式或安装必要的音频编解码器。错误详情: {str(e)}")

    def extract_mfcc_features(self) -> np.ndarray:
        """提取 MFCC 特征以及统计量"""
        # 1. MFCC统计特征
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr,
                                     n_mfcc=self.n_mfcc,
                                     n_fft=self.n_fft,
                                     hop_length=self.hop_length)
        # 1. 每帧的平均值（最重要的部分），音色基调
        mfcc_mean = np.mean(mfccs, axis=1)
        print(f"MFCC特征 mean 维度: {mfcc_mean.shape}, 前 4 个：{mfcc_mean[:4]}")
        # 标准差，音色变化程度
        mfcc_std = np.std(mfccs, axis=1)
        print(f"MFCC特征 std 维度: {mfcc_std.shape}, 前 4 个：{mfcc_std[:4]}")
        # 2. 全局统计（压缩信息）
        # 一阶差分: 表征MFCC的动态变化
        mfcc_delta = np.mean(librosa.feature.delta(mfccs), axis=1)
        print(f"MFCC特征 delta 维度: {mfcc_delta.shape}, 前 4 个：{mfcc_delta[:4]}")
        # 二阶差分: 表征变化的加速度
        mfcc_delta2 = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)
        print(f"MFCC特征 delta2 维度: {mfcc_delta2.shape}, 前 4 个：{mfcc_delta2[:4]}")
        # 合并：共80维（20×4）
        mfcc_all = np.concatenate([mfcc_mean, mfcc_std, mfcc_delta, mfcc_delta2])
        print(f"MFCC特征总维度: {mfcc_all.shape}")
        return mfcc_all

    def extract_spectral_features(self):
        """提取核心频谱特征"""
        spectral_features = []

        # 1. 频谱重心（Spectral Centroid） - 明亮度
        cent = librosa.feature.spectral_centroid(y=self.y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        spectral_features.append(np.mean(cent))  # 均值：整体明亮度
        spectral_features.append(np.std(cent))  # 标准差：明亮度变化
        print(f"频谱重心 mean: {spectral_features[0]:.4f}, std: {spectral_features[1]:.4f}")

        # 2. 频谱滚降（Spectral Rolloff） - 区分语音/音乐
        rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        spectral_features.append(np.mean(rolloff))  # 均值
        spectral_features.append(np.std(rolloff))  # 标准差
        print(f"频谱滚降 mean: {spectral_features[2]:.4f}, std: {spectral_features[3]:.4f}")

        # 3. 频谱带宽（Spectral Bandwidth） - 频谱宽度
        bandwidth = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr, n_fft=self.n_fft,
                                                       hop_length=self.hop_length)
        spectral_features.append(np.mean(bandwidth))  # 均值
        print(f"频谱带宽 mean: {spectral_features[4]:.4f}")

        # 4. 频谱对比度（Spectral Contrast） - 谐波突出度（可选）
        contrast = librosa.feature.spectral_contrast(
            y=self.y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            n_bands=self.n_bands
        )
        spectral_features.append(np.mean(contrast))  # 整体对比度
        print(f"频谱对比度 mean: {spectral_features[5]:.4f}")

        # 共 6 维
        features = np.array(spectral_features)
        print(f"频谱特征维度: {features.shape}")
        return features

    def extract_temporal_features(self):
        """提取时域特征"""
        temporal_features = []

        # 1. 过零率（Zero Crossing Rate） - 打击乐/持续音区分
        zcr = librosa.feature.zero_crossing_rate(self.y)
        temporal_features.append(np.mean(zcr))  # 均值：总体"尖锐度"
        temporal_features.append(np.std(zcr))  # 标准差：变化模式
        print(f"过零率 mean: {temporal_features[0]:.4f}, std: {temporal_features[1]:.4f}")

        # 2. 自相关峰值 - 节奏规律性（比直接RMS更有用）
        # 计算短时自相关找周期性
        autocorr = librosa.autocorrelate(self.y[:min(len(self.y), 44100)])  # 取前1秒
        if len(autocorr) > 10:
            # 找第一个显著峰值的位置（节奏周期）
            peaks = librosa.util.peak_pick(x=autocorr[10:], pre_max=10, post_max=10,
                                           pre_avg=10, post_avg=10, delta=0.5, wait=10)
            if len(peaks) > 0:
                temporal_features.append(peaks[0] / self.sr)  # 周期长度
            else:
                temporal_features.append(0.0)
        else:
            temporal_features.append(0.0)
        print(f"自相关峰值 mean: {temporal_features[2]:.4f}")

        # 共 3 维
        features = np.array(temporal_features)
        print(f"时域特征维度: {features.shape}")
        return features

    def extract_energy_features(self):
        """提取能量特征"""
        energy_features = []

        # 1. RMS能量 - 整体响度
        rms = librosa.feature.rms(y=self.y, frame_length=self.n_fft, hop_length=self.hop_length)
        energy_features.append(np.mean(rms))  # 均值：平均响度
        energy_features.append(np.std(rms))  # 标准差：动态范围
        energy_features.append(np.max(rms))  # 最大值：最强部分
        print(f"RMS特征 mean: {energy_features[0]:.4f}, std: {energy_features[1]:.4f}, max: {energy_features[2]:.4f}")

        # 2. 能量包络特征 - 歌曲结构
        # 计算能量上升/下降时间
        rms_norm = rms / (np.max(rms) + 1e-6)
        # 能量超过0.5的时间比例
        energy_above_half = np.sum(rms_norm > 0.5) / len(rms_norm)
        energy_features.append(energy_above_half)
        print(f"能量包络特征 mean: {energy_features[3]:.4f}")

        # 共 4 维
        features = np.array(energy_features)
        print(f"能量特征维度: {features.shape}")
        return features

    def extract_music_features(self):
        """提取音乐领域特征"""
        music_features = []

        # 1. 节奏（Tempo）- BPM
        tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr, hop_length=self.hop_length)
        music_features.append(tempo[0] if len(tempo) > 0 else 120)
        print(f"BPM: {music_features[0]:.4f}")

        # 2. 色度特征（Chroma） - 和声特征
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        # 取12个音级的平均值, 共 12 维
        chroma_mean = np.mean(chroma, axis=1)
        print(f"chroma_mean: {chroma_mean.shape}")
        music_features.extend(chroma_mean)
        print(f"色度特征 mean: {music_features[1]:.4f}, {music_features[2]:.4f}, ..., {music_features[12]:.4f}")

        # 3. 拍号（估计） - 节奏模式
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=self.sr)
        music_features.append(np.mean(pulse))  # 1维
        print(f"拍号特征 mean: {music_features[13]:.4f}")

        features = np.array(music_features)
        print(f"音乐特征维度: {features.shape}")
        return features

    def extract_all_features(self):
        """完整特征提取函数"""

        features = []

        # 2. 提取各类特征
        print("-" * 60 + " 1. 提取 MFCC 特征 " + "-" * 60)
        features.extend(self.extract_mfcc_features())  # 80维
        print("-" * 60 + " 2. 提取 频谱特征 " + "-" * 60)
        features.extend(self.extract_spectral_features())  # 6维
        print("-" * 60 + " 3. 提取 时域特征 " + "-" * 60)
        features.extend(self.extract_temporal_features())  # 3维
        print("-" * 60 + " 4. 提取 能量特征 " + "-" * 60)
        features.extend(self.extract_energy_features())  # 4维
        print("-" * 60 + " 5. 提取 音乐特征 " + "-" * 60)
        features.extend(self.extract_music_features())  # 14维

        # 3. 可选：简单元数据特征
        # 如果有标签信息，可以添加
        # features.extend(genre_one_hot)  # 例如流派独热编码

        # 总共约 107 维
        features = np.array(features)
        print(f"特征向量形状: {features.shape}")
        return features


if __name__ == '__main__':
    print("\n" + '=' * 120)
    audio_file = "../测试数据/小村庄月弯弯-晚月moon.mp3"
    # 创建特征提取器
    extractor = AudioFeatureExtractor(audio_file)
    # 提取特征
    features = extractor.extract_all_features()
    print(f'{audio_file} 特征向量形状: {features.shape}')
    print(f'{audio_file}\n'
          f'前 3 维: {json.dumps(features[:3], cls=CustomEncoder)}\n'
          f'后 3 维: {json.dumps(features[-3:], cls=CustomEncoder)}')

    print("\n" + '=' * 120)
    audio_file = "../测试数据/情火-洋澜一.m4a"
    extractor = AudioFeatureExtractor(audio_file)
    features = extractor.extract_all_features()
    print(f'{audio_file} 特征向量形状: {features.shape}')
    print(f'{audio_file}\n'
          f'前 3 维: {json.dumps(features[:3], cls=CustomEncoder)}\n'
          f'后 3 维: {json.dumps(features[-3:], cls=CustomEncoder)}')
