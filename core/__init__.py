import torch
from diffusers import DiffusionPipeline

_PIPE = None

def get_pipeline():
    global _PIPE
    if _PIPE is None:
        raise Exception("model not initialized")
    _PIPE.scheduler = _PIPE.scheduler.from_config(_PIPE.scheduler.config)
    # p = DiffusionPipeline.from_pipe(_PIPE, scheduler=_PIPE.scheduler.from_config(_PIPE.scheduler.config))
    return _PIPE

def init_pipeline(args):
    global _PIPE
    _PIPE = DiffusionPipeline.from_pretrained(args.model).to(args.device)
