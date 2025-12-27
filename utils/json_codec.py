import json
from datetime import datetime
from enum import Enum

import numpy as np
from flask.json.provider import DefaultJSONProvider


def codec(obj):
    """ 自定义编解码器，特殊对象转换 """
    if isinstance(obj, datetime):
        # 处理日期时间对象
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(obj, Enum):
        # 处理枚举对象
        return obj.name
    elif isinstance(obj, np.ndarray):
        # 处理容器类型中的特殊对象
        return obj.tolist()
    elif isinstance(obj, np.integer):
        # 处理numpy整数类型
        return int(obj)
    elif isinstance(obj, np.floating):
        # 处理numpy浮点数类型
        return float(obj)
    return None


class CustomJSONProvider(DefaultJSONProvider):
    """ 自定义编解码器, 用于flask """

    def default(self, obj):
        """处理特殊对象"""
        codec_result = codec(obj)
        return codec_result if codec_result else super(CustomJSONProvider, self).default(obj)


class CustomEncoder(json.JSONEncoder):
    """ 自定义编解码器，用于 json """

    def default(self, obj):
        codec_result = codec(obj)
        return codec_result if codec_result else super(CustomEncoder, self).default(obj)
