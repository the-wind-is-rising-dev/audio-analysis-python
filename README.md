# audio-analysis-python

音频分析 Python 工具包，提供全面的音频特征提取、歌词分析和相似度计算功能，并支持 REST API 服务。

## 功能特性

### 音频特征提取

- **MFCC 特征**：提取 20 维 MFCC 系数及其一阶、二阶差分，共 80 维特征
- **频谱特征**：频谱重心、滚降、带宽、对比度等 6 维特征
- **时域特征**：过零率、自相关峰值等 3 维特征
- **能量特征**：RMS 能量、动态范围等 4 维特征
- **音乐特征**：BPM、色度特征、拍号等 14 维特征
- **完整特征集**：整合所有特征，共 107 维向量

### 歌词分析

- 使用轻量级 BERT 模型（paraphrase-MiniLM-L6-v2）获取歌词语义向量
- 支持将 384 维语义向量降维至目标维度
- 支持批量处理歌词关键词

### 特征标准化

- 基于 sklearn 实现特征标准化
- 支持单个特征和批量特征的标准化处理
- 支持标准化器的保存和加载
- 提供 PCA 特征降维功能

### 相似度计算

- 基于余弦相似度算法

### Web 服务

- 基于 Flask 框架的 REST API
- 支持跨域请求
- 完整的请求/响应日志
- 自定义 JSON 编解码器

## 项目结构

```
audio-analysis-python/
├── core/                 # 核心功能模块
│   ├── __init__.py
│   ├── audio_features.py   # 音频特征提取
│   ├── lyric_analyzer.py   # 歌词分析
│   └── standardization.py  # 标准化处理
├── server/               # Web服务模块
│   ├── config/           # 配置文件
│   ├── controller/       # API控制器
│   ├── service/          # 业务逻辑
│   └── application.py    # Flask应用入口
├── similarity/           # 相似度计算
│   └── cosine_similarity.py
├── utils/                # 工具类
│   └── json_codec.py
├── 测试数据/              # 测试数据
│   ├── standardizer.pkl
│   ├── 小村庄月弯弯-晚月moon.mp3
│   ├── 情火-洋澜一.m4a
│   └── 歌词关键词列表.json
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 安装

1. 克隆项目

```bash
git clone git@github.com:the-wind-is-rising-dev/audio-analysis-python.git
cd audio-analysis-python
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 音频特征提取

```python
from core.audio_features import AudioFeatureExtractor

# 示例1：提取单个音频文件的所有特征
audio_file = "../测试数据/小村庄月弯弯-晚月moon.mp3"
# 创建特征提取器
extractor = AudioFeatureExtractor(audio_file)
# 提取完整特征
features = extractor.extract_all_features()
print(f'{audio_file} 特征向量形状: {features.shape}')
print(f'{audio_file} 前 3 维: {features[:3]}')
print(f'{audio_file} 后 3 维: {features[-3:]}')

# 示例2：提取多个音频文件的特征
audio_files = [
    "../测试数据/小村庄月弯弯-晚月moon.mp3",
    "../测试数据/情火-洋澜一.m4a"
]

for file in audio_files:
    extractor = AudioFeatureExtractor(file)
    features = extractor.extract_all_features()
    print(f'{file} 特征向量形状: {features.shape}')
    print(f'{file} 前 3 维: {features[:3]}')
    print(f'{file} 后 3 维: {features[-3:]}')

# 示例3：提取特定类型特征
extractor = AudioFeatureExtractor(audio_file)
mfcc_features = extractor.extract_mfcc_features()
print(f'MFCC特征维度: {mfcc_features.shape}')

spectral_features = extractor.extract_spectral_features()
print(f'频谱特征维度: {spectral_features.shape}')

temporal_features = extractor.extract_temporal_features()
print(f'时域特征维度: {temporal_features.shape}')

energy_features = extractor.extract_energy_features()
print(f'能量特征维度: {energy_features.shape}')

music_features = extractor.extract_music_features()
print(f'音乐特征维度: {music_features.shape}')
```

### 2. 歌词分析

```python
from core.lyric_analyzer import LyricsProcessor
import json

# 示例1：从文件加载歌词关键词并进行分析
max_features = 86

# 从JSON文件加载歌词关键词列表
with open('../测试数据/歌词关键词列表.json', 'r', encoding='utf-8') as f:
    lyrics_keywords_list = json.load(f)

print("歌词关键词特征批量获取")
processor = LyricsProcessor()
lyrics_embeddings = processor.bert_encode(lyrics_keywords_list)
print(f"歌词特征维度: {lyrics_embeddings.shape}")

print("\n降维")
lyrics_reduced = processor.reduce_lyrics_dimension(lyrics_embeddings, target_dim=max_features)
print(f"降维后的歌词特征维度: {lyrics_reduced.shape}")

# 示例2：直接分析歌词关键词列表
lyrics_keywords = ["爱情", "悲伤", "回忆", "快乐", "希望"]
processor = LyricsProcessor()
embeddings = processor.bert_encode(lyrics_keywords)
print(f"直接分析关键词 - 特征维度: {embeddings.shape}")
reduced = processor.reduce_lyrics_dimension(embeddings, target_dim=86)
print(f"直接分析关键词 - 降维后维度: {reduced.shape}")
```

### 3. 特征标准化器

```python
from core.standardization import Standardizer
from core.lyric_analyzer import LyricsProcessor
import json
import numpy as np

# 示例1：初始化标准化器并保存
max_features = 86

# 获取歌词特征作为示例数据
with open('../测试数据/歌词关键词列表.json', 'r', encoding='utf-8') as f:
    lyrics_keywords_list = json.load(f)

processor = LyricsProcessor()
lyrics_embeddings = processor.bert_encode(lyrics_keywords_list)
lyrics_reduced = processor.reduce_lyrics_dimension(lyrics_embeddings, target_dim=max_features)

# 初始化并训练标准化器
scaler_path = "../测试数据/standardizer.pkl"
standardizer = Standardizer(scaler_path)
features_scaled = standardizer.init_standardizer(np.array(lyrics_reduced))
print(f"标准化后的特征维度: {features_scaled.shape}")

# 保存标准化器
standardizer.save()
print(f"标准化器已保存到: {scaler_path}")

# 示例2：加载已有标准化器并处理新特征
new_lyrics_keywords = ["新歌", "流行", "动感"]
new_embeddings = processor.bert_encode(new_lyrics_keywords)
new_reduced = processor.reduce_lyrics_dimension(new_embeddings, target_dim=max_features)

# 加载已有标准化器
loaded_standardizer = Standardizer(scaler_path)

# 处理单个新特征
scaled_feature = loaded_standardizer.process_new_features(new_reduced[0])
print(f"单个特征标准化后维度: {scaled_feature.shape}")

# 批量处理新特征
scaled_features = loaded_standardizer.batch_process_new_features(new_reduced)
print(f"批量特征标准化后维度: {scaled_features.shape}")
```

### 4. 余弦相似度计算

```python
from similarity.cosine_similarity import cosine_similarity, one_to_more_cosine_similarity
import numpy as np

# 示例1：计算两个向量之间的余弦相似度
print('=' * 40 + ' 测试向量余弦相似度 ' + '=' * 40)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
similarity = cosine_similarity(a, b)
print(f'similarity: {similarity}')

# 示例2：计算一个向量与多个向量的余弦相似度
print('\n' + '=' * 40 + ' 测试矩阵余弦相似度 ' + '=' * 40)
source_vec = np.array([1, 2, 3])
target_vec = np.array([[7, 8, 9], [10, 11, 12]])
similarities = one_to_more_cosine_similarity(source_vec, target_vec)
print(f'similarities: {similarities}')
```

### 5. 启动 Web 服务

```bash
python server/application.py
```

服务将在 `http://0.0.0.0:5001` 启动，支持以下 API 端点：

- `/numpy` - NumPy 相关功能
- `/feature-extraction` - 音频特征提取

## API 文档

### 音频特征提取 API

```bash
curl --location --request POST 'http://127.0.0.1:5001/feature-extraction/audio-feature-extraction' \
--header 'Content-Type: application/json' \
--data-raw '{
    "audioFilepath": "情火-洋澜一.m4a"
}'
```

### 提取歌词特征 API

```bash
curl --location --request POST 'http://127.0.0.1:5001/feature-extraction/lyrics-feature-extraction' \
--header 'Content-Type: application/json' \
--data-raw '{
    "lyricsList": [
        "爱 宿命 仙魔 善恶 灵魂 天命 缠绵 坠爱 万丈 大雨 滂沱 风月 醉眼 弹指间 爱恨 幻灭 无渊 波澜 时光 万劫不复 纠缠 生生世世",
        "记忆重现 失去 守护 恳求 惶惶不安 难以自制 看着 记得 暂时留步 消失"
    ]
}'
```

### 初始化特征标准化器 API

```bash
curl --location --request POST 'http://127.0.0.1:5001/feature-extraction/init-feature-standardizer' \
--header 'Content-Type: application/json' \
--data-raw '{
    "scalerFilepath": "standardizer.pkl",
    "featuresMatrixFilepath": "features_matrix.json",
    "featuresScaledFilepath": "scaled_filepath.json"
}'
```

### 根据已有标准化器处理新特征 API

```bash
curl --location --request POST 'http://127.0.0.1:5001/feature-extraction/feature-standardization' \
--header 'Content-Type: application/json' \
--data-raw '{
  "scaler_filepath": "standardizer.pkl",
  "features_matrix": [
    [
      1.0,
      2.0,
      3.0
    ]
  ]
}'
```

### 计算向量算数平均值 API

```bash
curl --location --request POST 'http://127.0.0.1:5001/numpy/mean_axis_0' \
--header 'Content-Type: application/json' \
--data-raw '[
    [62],
    [3]
]'
```

### 计算两个向量之间的余弦相似度 API

```bash
curl --location --request POST 'http://127.0.0.1:5001/numpy/cosine_similarity' \
--header 'Content-Type: application/json' \
--data-raw '[
    [62,0],
    [11,23]
]'
```

### 计算一个向量与多个向量的余弦相似度 API

```bash
curl --location --request POST 'http://127.0.0.1:5001/numpy/one_to_more_cosine_similarity' \
--header 'Content-Type: application/json' \
--data-raw '{
    "source": [63,27],
    "targetList": [
        [32,25],
        [32,-10]
    ]
}'
```

## 测试数据

项目包含以下测试数据：

- `小村庄月弯弯-晚月moon.mp3` - 测试音频文件 1
- `情火-洋澜一.m4a` - 测试音频文件 2
- `歌词关键词列表.json` - 测试歌词关键词
- `standardizer.pkl` - 标准化模型

## 技术栈

- **Python** - 主要开发语言
- **librosa** - 音频处理库
- **sentence-transformers** - BERT 模型实现
- **scikit-learn** - PCA 降维
- **Flask** - Web 框架
- **numpy** - 数值计算
- **cosine-similarity** - 相似度计算

## 许可证

详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，欢迎通过 GitHub Issues 反馈。
