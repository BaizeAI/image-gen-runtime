from flask import Blueprint, request, jsonify
import base64
import logging
import torch
import uuid
from io import BytesIO
import threading
from models import GenerateImageRequest, Image, GenerateImageResponse
from core import get_pipeline

class ImageGenerationAPI:
    stop_events = {}
    def __init__(self):
        self.api_bp = Blueprint('api', __name__)
        self.stop_events = {}
        self.should_exit = False

        # Register routes
        self.api_bp.add_url_rule('/v1/images/generations', 'generate_image', self.generate_image, methods=['POST'])
        self.api_bp.add_url_rule('/health', 'health', self.healthz, methods=['GET'])

        # Register error handlers
        self.api_bp.register_error_handler(404, self.page_not_found)
        self.api_bp.register_error_handler(Exception, self.handle_exception)

    def page_not_found(self, e):
        return jsonify({
            "error": "Resource not found",
            "message": "The requested URL was not found on the server."
        }), 404

    def handle_exception(self, e):
        logging.error(f"Error occurred: {e.__class__.__name__}, Message: {str(e)}", exc_info=True)
        if isinstance(e, (ValueError, AssertionError)):
            code = 400
        else:
            code = 500
            self.should_exit = True
        return jsonify({
            "error": e.__class__.__name__,
            "message": str(e)
        }), code

    async def generate_image(self):
        data = request.json or {}
        req = GenerateImageRequest(**data)
        req.validate()
        logging.debug(f'received request: {req=}')
        request_id = str(uuid.uuid4())
        self.stop_events[request_id] = threading.Event()
        # todo 解决当客户端取消请求之后，Pipeline 不会结束的问题
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
            images = resp.images
        # try:
        #     with torch.no_grad():
        #         resp = await asyncio.to_thread(
        #             get_pipeline(),
        #             req.prompt,
        #             height=req.height,
        #             width=req.width,
        #             negative_prompt=req.negative_prompt,
        #             num_images_per_prompt=req.n,
        #             num_inference_steps=req.num_inference_steps,
        #             guidance_scale=req.guidance_scale,
        #             callback_on_step_end=self.get_cancelable_callback(request_id),
        #         )
        #         images = resp.images
        # except RuntimeError as e:
        #     if "User canceled generation" in str(e):
        #         return jsonify({"status": "canceled", "request_id": request_id})
        #     raise e

        _images = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            _images.append(Image(b64_json=img_str))

        response_data = GenerateImageResponse(data=_images)
        return jsonify(response_data.as_dict())

    def healthz(self):
        if self.should_exit:
            return 'Fail', 500
        assert get_pipeline() is not None
        return 'OK\n'

    def get_blueprint(self):
        return self.api_bp
