from flask import Blueprint, request, jsonify
import base64
import logging
import torch
import uuid
from io import BytesIO
import threading
import queue
from concurrent.futures import Future
import atexit

from models import GenerateImageRequest, Image, GenerateImageResponse
from core import get_pipeline


class ImageGenerationAPI:
    """REST API —— 顺序（非并发）图片生成服务。

    原理：
        - 主线程（HTTP 视图）只负责把请求放进队列并同步等待结果；
        - 后台 worker 线程串行消费队列，确保同一时间只有一个生成任务在跑。
        - 若队列已满直接返回 429，避免无限堆积。
    """

    # 全局任务队列（可按需调大 / 调小 maxsize）
    _job_q: "queue.Queue[tuple[GenerateImageRequest, Future]]" = queue.Queue(maxsize=32)

    def __init__(self):
        # Flask Blueprint
        self.api_bp = Blueprint('api', __name__)
        self.should_exit = False

        self.api_bp.add_url_rule('/v1/images/generations', 'generate_image', self.generate_image, methods=['POST'])
        self.api_bp.add_url_rule('/health', 'health', self.healthz, methods=['GET'])

        self.api_bp.register_error_handler(404, self.page_not_found)
        self.api_bp.register_error_handler(Exception, self.handle_exception)

        self._worker = threading.Thread(target=self._worker_loop, name="image-generator-worker", daemon=True)
        self._worker.start()
        atexit.register(self.shutdown)  # 进程退出前优雅关停

    def page_not_found(self, e):
        return jsonify({
            "error": "Resource not found",
            "message": "The requested URL was not found on the server."
        }), 404

    def handle_exception(self, e):
        logging.error(f"Error occurred: {e.__class__.__name__}, Message: {str(e)}", exc_info=True)
        code = 400 if isinstance(e, (ValueError, AssertionError)) else 500
        if code == 500:
            self.should_exit = True
        return jsonify({
            "error": e.__class__.__name__,
            "message": str(e)
        }), code

    def generate_image(self):
        """POST /v1/images/generations

        接收生成请求 -> 丢进队列 -> 阻塞等待 worker 返回结果。
        若队列已满返回 429。"""
        data = request.json or {}
        req = GenerateImageRequest(**data)
        req.validate()
        logging.debug(f"received request: {req=}")

        fut: Future = Future()
        try:
            # 若队列已满抛 queue.Full
            self._job_q.put_nowait((req, fut))
        except queue.Full:
            return jsonify({
                "error": "queue_full",
                "message": "Server busy, try again later."
            }), 429

        # 同步等待结果 / 异常
        try:
            images = fut.result()  # 阻塞直到 worker 调用 set_result / set_exception
        except Exception as e:
            # 把异常继续交给统一处理
            raise e

        # 编码并返回
        _images = []
        for image in images:
            buf = BytesIO()
            image.save(buf, format="PNG")
            _images.append(Image(b64_json=base64.b64encode(buf.getvalue()).decode()))

        return jsonify(GenerateImageResponse(data=_images).as_dict())

    def healthz(self):
        if self.should_exit:
            return 'Fail', 500
        assert get_pipeline() is not None
        return 'OK\n'

    def get_blueprint(self):
        return self.api_bp

    @staticmethod
    def _do_generate(req: GenerateImageRequest):
        """真正的图片生成逻辑（与原代码保持一致）。"""
        with torch.no_grad():
            pipe = get_pipeline()
            resp = pipe(
                req.prompt,
                height=req.height,
                width=req.width,
                negative_prompt=req.negative_prompt,
                num_images_per_prompt=req.n,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
            )
        return resp.images

    def _worker_loop(self):
        """后台线程：顺序消费任务队列。"""
        while True:
            item = self._job_q.get()  # 阻塞等待
            if item is None:
                break  # 收到毒丸退出
            req, fut = item
            try:
                images = self._do_generate(req)
                fut.set_result(images)
            except Exception as e:
                fut.set_exception(e)
            finally:
                self._job_q.task_done()

    def shutdown(self):
        """进程退出时优雅关闭 worker。"""
        try:
            self._job_q.put_nowait(None)
        except queue.Full:
            pass
        if self._worker.is_alive():
            self._worker.join(timeout=5)
