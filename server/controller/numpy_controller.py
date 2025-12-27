"""
numpy 计算
"""
import numpy as np
from flask import Blueprint, request

from server.controller.result import Result
from similarity.cosine_similarity import cosine_similarity, one_to_more_cosine_similarity

"""创建蓝图"""
numpy_bp = Blueprint("numpy", __name__, url_prefix="/numpy")


@numpy_bp.post("/mean_axis_0")
def mean_axis_0():
    """计算向量算术平均值"""
    # 获取输入向量
    body = request.json
    if not isinstance(body, list) or len(body) < 2:
        return Result.fail(message="请最少输入两个向量")
    a = np.array(body).astype(np.float64)
    # 计算向量平均值
    average = np.mean(a, axis=0)
    return Result.success(result=average)


@numpy_bp.post("/cosine_similarity")
def cosine_similarity_controller():
    """计算两个向量之间的余弦相似度"""

    body = request.json
    if not isinstance(body, list) or len(body) != 2:
        return Result.fail(message="请输入两个向量")

    # 计算余弦相似度
    similarity = cosine_similarity(body[0], body[1])
    return Result.success(result=similarity)


@numpy_bp.post("/one_to_more_cosine_similarity")
def one_to_more_cosine_similarity_func():
    """ 计算一个向量与多个向量的余弦相似度"""
    body = request.json
    source = body["source"]
    target_list = body["targetList"]
    if not source or not target_list:
        return Result.fail(message="请输入源向量和目标向量")

    # 向量化转换
    source_vec = np.asarray(source, dtype=np.float64)
    targets_mat = np.asarray(target_list, dtype=np.float64)

    # 初始化结果数组
    similarities = one_to_more_cosine_similarity(source_vec, targets_mat)

    return Result.success(result=similarities.tolist())
