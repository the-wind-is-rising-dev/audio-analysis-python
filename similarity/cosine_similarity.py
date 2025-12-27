import numpy as np


def cosine_similarity(a, b):
    """
    计算两个向量之间的余弦相似度

    参数:
    a, b : array_like
        输入的两个向量

    返回:
    float
        两向量间的余弦相似度值，范围 [-1, 1]
    """
    # 确保输入为 numpy 数组
    if not isinstance(a, np.ndarray):
        a = np.array(a, dtype=np.float64)
    if not isinstance(b, np.ndarray):
        b = np.array(b, dtype=np.float64)
    # 检查维度是否匹配
    if a.shape != b.shape:
        raise ValueError("向量维度不匹配")
    # 计算点积
    dot_product = np.dot(a, b)

    # 计算各向量的模长（L2范数）
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # 避免除零错误
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # 计算余弦相似度
    similarity = dot_product / (norm_a * norm_b)

    return similarity


def one_to_more_cosine_similarity(source_vec: np.ndarray, targets_mat: np.ndarray):
    """
    计算一个向量与多个向量的余弦相似度
    :param source_vec:  源向量 (n,)
    :param targets_mat: 目标向量矩阵(m,n)
    :return:
    """

    # 快速检查维度
    if source_vec.ndim != 1:
        raise Exception("源向量维度错误，源向量必须是(n,)")

    if targets_mat.ndim != 2 or targets_mat.shape[1] != source_vec.shape[0]:
        raise Exception("目标向量矩阵维度错误, 目标向量矩阵必须为(m,n)")

    # 计算模长
    source_norm = np.linalg.norm(source_vec)

    # 特殊情况：源向量为零向量
    if source_norm == 0:
        return np.zeros(targets_mat.shape[0])

    # 向量化计算点积
    dot_products = np.dot(targets_mat, source_vec)

    # 计算目标向量的模长
    targets_norm = np.linalg.norm(targets_mat, axis=1)

    # 避免除零（处理目标向量的零向量）
    EPSILON = 1e-10
    denominator = source_norm * targets_norm
    mask = denominator > EPSILON

    # 初始化结果数组
    similarities = np.zeros(targets_mat.shape[0])
    similarities[mask] = dot_products[mask] / denominator[mask]

    return similarities


if __name__ == '__main__':
    print('\n' + '=' * 40 + ' 测试向量余弦相似度 ' + '=' * 40)
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    similarity = cosine_similarity(a, b)
    print(f'similarity:{similarity}')

    print('\n' + '=' * 40 + ' 测试矩阵余弦相似度 ' + '=' * 40)
    source_vec = np.array([1, 2, 3])
    target_vec = np.array([[7, 8, 9], [10, 11, 12]])
    similarities = one_to_more_cosine_similarity(source_vec, target_vec)
    print(f'similarities:{similarities}')
