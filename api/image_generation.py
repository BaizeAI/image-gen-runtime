from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import base64
import logging
import torch
import uuid
from io import BytesIO
import threading
from typing import Optional
import asyncio
import functools

from models import GenerateImageRequest, Image, GenerateImageResponse
from core import get_pipeline


async def listen_for_disconnect(request: Request) -> None:
    """Returns if a disconnect message is received"""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    # Functools.wraps is required for this wrapper to appear to fastapi as a
    # normal route handler, with the correct request type hinting.
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):

        # The request is either the second positional arg or `raw_request`
        request = args[1] if len(args) > 1 else kwargs["raw_request"]

        cancellation_task = asyncio.create_task(listen_for_disconnect(request))
        handler_task = asyncio.create_task(handler_func(*args, **kwargs))

        done, pending = await asyncio.wait([handler_task, cancellation_task],
                                           return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()

        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


class ImageGenerationAPI:
    def __init__(self):
        self.router = APIRouter()
        self.should_exit = False

        self.router.add_api_route('/v1/images/generations', self.generate_image, methods=['POST'])
        self.router.add_api_route('/health', self.healthz, methods=['GET'])
        self.lock = asyncio.Lock()


    @with_cancellation
    async def generate_image(self, request: GenerateImageRequest, raw_request: Request):
        req = request
        req.validate()
        logging.debug(f"received request: {req=}")
        cancel_ev = threading.Event()
        async with self.lock:
            try:
                images = await asyncio.to_thread(self._do_generate, req, cancel_ev)
                _images = []
                for image in images:
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    _images.append(Image(b64_json=img_str))

                response_data = GenerateImageResponse(data=_images)
                return response_data.as_dict()
            except asyncio.CancelledError:
                logging.info(f'req({req.id=}) aborted')
                cancel_ev.set()

    async def healthz(self):
        if self.should_exit:
            raise HTTPException(status_code=500, detail="Fail")
        assert get_pipeline() is not None
        return "OK"

    def get_router(self):
        return self.router

    @staticmethod
    def _do_generate(req: GenerateImageRequest, cancel_event: threading.Event):
        """真正的图片生成逻辑（与原代码保持一致）。"""
        def callback(pipeline, i, t, callback_kwargs):
            if cancel_event.is_set():
                pipeline._interrupt = True
            return callback_kwargs
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
                callback_on_step_end=callback,
            )
        return resp.images
