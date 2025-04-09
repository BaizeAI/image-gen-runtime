from diffusers import DiffusionPipeline
import traceback

_SHARED_PIPE = None

def get_pipeline():
    global _SHARED_PIPE
    if _SHARED_PIPE is None:
        raise Exception("model not initialized")
    # create a copy of share pipe to fix the currency issue https://github.com/huggingface/diffusers/issues/3672
    p = _SHARED_PIPE.from_pipe(pipeline=_SHARED_PIPE, scheduler=_SHARED_PIPE.scheduler.from_config(_SHARED_PIPE.scheduler.config))
    return p

def init_pipeline(args):
    global _SHARED_PIPE
    if args.custom_load_pipe_script:
        try:
            out = {}
            exec(args.custom_load_pipe_script, {}, out)
            _SHARED_PIPE = out.get('pipe', None)
        except Exception:
            traceback.print_exc()
    else:
        _SHARED_PIPE = DiffusionPipeline.from_pretrained(args.model, scheduler=scheduler).to(args.device)
    assert isinstance(_SHARED_PIPE, DiffusionPipeline), "pipeline init error, don't forget to assign pipeline to pipe var"
