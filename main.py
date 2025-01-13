import argparse
import time
import torch
import base64
import asyncio

from flask import Flask, request, jsonify
from diffusers import DiffusionPipeline
from io import BytesIO
from typing import List
from dataclasses import dataclass, asdict, field

app = Flask(__name__)

@app.errorhandler(404)
def page_not_found(e):
    # 可以返回字符串、JSON 或者渲染模板
    return jsonify({
        "error": "Resource not found",
        "message": "The requested URL was not found on the server."
    }), 404

@app.errorhandler(Exception)
def page_not_found(e):
    code = 500
    if isinstance(e, (ValueError, AssertionError)):
        code = 400
    # todo others code
    return jsonify({
        "error": e.__class__.__name__,
        "message": str(e)
    }), code

@dataclass
class GenerateImageRequest:
    prompt: str
    n: int = 1
    quality: str = 'hd'
    size: str = '512x512'
    model: str = ''
    response_format: str = 'b64_json'

    negative_prompt: str = ''
    guidance_scale: float = 7.5

    def validate(self):
        ps = self.size.split('x')
        if len(ps) != 2:
            raise ValueError(f'{self.size} is not a valid size')
        if int(ps[0]) % 8 != 0 or int(ps[1]) % 8 != 0:
            raise ValueError(f'size width or height must be multiples of 8')
        if self.quality not in ('hd',):
            raise ValueError(f'{self.quality} is not a valid quality')
        assert self.num_inference_steps >= 1, "num_inference_steps must be greater than or equal to 1"
        assert 1 <= self.n <= 9, "n must be between 1 and 9"
        # todo impl url
        assert self.response_format in ('b64_json', ), "response_format must be 'b64_json' or 'url'"

    @property
    def num_inference_steps(self):
        m = {
            'hd': 50
        }
        return m[self.quality]

    @property
    def width(self):
        ps = self.size.split('x')
        return int(ps[0])

    @property
    def height(self):
        ps = self.size.split('x')
        return int(ps[1])


@dataclass
class Image:
    b64_json: str = None
    url: str = None
    revised_prompt: str = None


@dataclass
class GenerateImageResponse:
    data: List[Image]
    created: int = field(default_factory=lambda: time.time())

@app.route('/v1/images/generations', methods=['POST'])
async def generate_image():
    data = request.json or {}
    req = GenerateImageRequest(**data)
    req.validate()

    with torch.no_grad():
        resp = await asyncio.to_thread(
            get_pipeline(),
            req.prompt,
            height=req.height,
            width=req.width,
            negative_prompt=req.negative_prompt,
            num_images_per_prompt=req.n,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale
        )
        images = resp.images

    _images = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = 'data:image/png;base64,'+base64.b64encode(buffered.getvalue()).decode('utf-8')
        _images.append(Image(b64_json=img_str))

    response_data = GenerateImageResponse(data=_images)
    return jsonify(asdict(response_data))

@app.route('/healthz', methods=['GET'])
def healthz():
    assert get_pipeline() is not None
    return 'OK\n'

_PIPE = None

def get_pipeline():
    global _PIPE
    if _PIPE is None:
        raise Exception("model not initialized")
    return _PIPE

def init_pipeline(args):
    global _PIPE
    _PIPE = DiffusionPipeline.from_pretrained(args.model).to(args.device)

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion Image Generation Service")
    parser.add_argument(
        "--model",
        type=str,
        default="sd-legacy/stable-diffusion-v1-5",
        help="HuggingFace model ID or local path for the DiffusionPipeline."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for Flask to listen on."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for Flask to listen on."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run the model on. 'auto' will automatically choose CUDA if available, otherwise CPU."
    )
    # 其他可选参数，你也可以自行添加，比如默认的 n、quality、size 等
    # parser.add_argument("--default_size", type=str, default="512x512", help="Default generation size, e.g. '512x512'")

    args = parser.parse_args()

    # 如果用户选择 device=auto，则自动判断 CUDA 是否可用
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args

if __name__ == '__main__':
    args = parse_args()

    print(f"Loading model {args.model} on device {args.device}...")
    init_pipeline(args)
    print("Done.")

    # 启动 Flask 服务
    app.run(host=args.host, port=args.port, threaded=True)
