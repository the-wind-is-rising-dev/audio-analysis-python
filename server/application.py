import json
import platform
import threading
import traceback
import typing as t
from datetime import datetime

from flask import Flask, Blueprint, request
from flask_cors import CORS

from config.log_config import log_config
from controller.result import Result
from utils.json_codec import CustomJSONProvider


class FlaskApp(Flask):
    def __init__(self, import_name: str, log_filename: str, **kwargs):
        super().__init__(import_name, **kwargs)
        # 日志配置
        log_config(name=import_name, log_filename=log_filename)

        # 注册蓝图
        self.blueprints_to_register = []
        from controller.test_controller import test_bp
        self.add_blueprint(bp=test_bp)

        # 添加请求前和响应后处理逻辑
        self.before_request(self.log_request_info)
        self.after_request(self.process_response)

        # 添加错误处理
        self.errorhandler(Exception)(self.handle_exception)

        # 请求耗时统计
        self.request_duration_local = threading.local()

        # 自定义编解码器
        self.json = CustomJSONProvider(self)

    def log_request_info(self):
        """记录请求信息"""
        self.request_duration_local.start_time = datetime.now()
        path_log = f'{"=" * 40} 请求路径: {request.path} {"=" * 40}'
        headers_log = '\n'.join([f'{key}: {value}' for key, value in request.headers.items()])
        request_body = request.get_json() if request.is_json else None
        request_body_str = json.dumps(request_body, ensure_ascii=False)
        if len(request_body_str) > 1024:
            request_body_str = f'{request_body_str[0:256]}...{request_body_str[-256:]}'
        print(
            f'请求信息\n'
            f'{path_log}\n'
            f'请求方法: {request.method}\n'
            f'{headers_log}\n'
            f'请求参数/args: {json.dumps(request.args, ensure_ascii=False)}\n'
            f'请求体/json: {request_body_str}\n'
            f'{"=" * (len(path_log) + 3)}')

    def process_response(self, response):
        """记录响应信息"""
        duration = datetime.now() - self.request_duration_local.start_time
        path_log = f'{"=" * 40} 响应信息: {request.path} {"=" * 40}'
        if response.is_json:
            resp_result = json.dumps(response.get_json(), ensure_ascii=False)
        else:
            resp_result = response.data
        if len(resp_result) > 1024:
            resp_result = f'{resp_result[0:256]}...{resp_result[-256:]}'
        print(
            f'响应信息\n'
            f'{path_log}\n'
            f'请求路径: {request.path} 请求方法: {request.method}\n'
            f'响应状态码: {response.status_code} 耗时: {duration.total_seconds():.3f}s\n'
            f'响应结果: {resp_result}\n'
            f'{"=" * (len(path_log) + 3)}')
        return response

    def handle_exception(self, ex: Exception):
        """处理未捕获的异常"""
        traceback.print_exc()
        # 返回自定义错误响应
        status_code = 500
        if '404 Not Found' in str(ex):
            status_code = 404
        return Result.fail(f'{ex}'), status_code

    def add_blueprint(self, bp: Blueprint):
        """添加需要注册的蓝图"""
        self.blueprints_to_register.append(bp)

    def set_cors_and_register_blueprint(self, bp: Blueprint, cors_config: dict = None):
        """
        配置跨域并注册蓝图
        """
        if cors_config is None:
            cors_config = {
                "origins": "*",
                "allow_headers": ["Content-Type"],
                "methods": ["GET", "PUT", "POST", "DELETE", "OPTIONS", "PATCH"]
            }

        # 跨域配置
        print(f'跨域配置: {bp.name}-{json.dumps(cors_config)}')
        CORS(bp, **cors_config)
        # 注册蓝图
        print(f'注册蓝图: {bp.name}')
        self.register_blueprint(bp)

    def run(self, host: str | None = None, port: int | None = None,
            debug: bool | None = None, load_dotenv: bool = True, **options: t.Any) -> None:

        for bp in self.blueprints_to_register:
            self.set_cors_and_register_blueprint(bp)

        super().run(host=host, port=port, debug=debug, load_dotenv=load_dotenv, **options)


if __name__ == "__main__":
    log_filename = './python.log'
    app = FlaskApp(__name__, log_filename)
    app.run(host="0.0.0.0", port=5001, debug=platform.system() == 'Darwin')
