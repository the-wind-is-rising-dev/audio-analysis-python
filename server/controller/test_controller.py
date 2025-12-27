from datetime import datetime

import numpy as np
from flask import Blueprint, request

from server.controller.result import Result

"""
创建蓝图
"""
test_bp = Blueprint("test", __name__, url_prefix="/test")


@test_bp.post("post")
def test_post():
    """
    测试post请求
    """
    request_body = request.json
    data = {
        "request_body": request_body,
        "request_time": datetime.now(),
        "ndarray": np.array([1, 2, 3]),
    }
    print(data)
    return Result.success(data)


@test_bp.get("current_time")
def current_time():
    """
    测试get请求
    """
    args = request.args
    data = {
        "args": args,
        "request_time": datetime.now(),
    }
    print(data)
    return Result.success(data)


@test_bp.get("error")
def test_error():
    """
    测试异常
    """
    raise Exception("测试异常")
