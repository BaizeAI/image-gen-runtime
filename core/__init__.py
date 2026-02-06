from diffusers import DiffusionPipeline
import cache_dit
import traceback
from config import get_config
from metrics import track_pipeline_load

_SHARED_PIPE = None

def get_pipeline():
    global _SHARED_PIPE
    if _SHARED_PIPE is None:
        raise Exception("model not initialized")
    if get_config().duplicate_scheduler:
        # create a copy of share pipe to fix the currency issue https://github.com/huggingface/diffusers/issues/3672
        p = _SHARED_PIPE.from_pipe(pipeline=_SHARED_PIPE, scheduler=_SHARED_PIPE.scheduler.from_config(_SHARED_PIPE.scheduler.config))
    else:
        p = _SHARED_PIPE
    return p

@track_pipeline_load
def init_pipeline(args):
    global _SHARED_PIPE
    if args.custom_load_pipe_script:
        try:
            out = {}
            exec(args.custom_load_pipe_script, {
                'args': args,
            }, out)
            _SHARED_PIPE = out.get('pipe', None)
        except Exception:
            traceback.print_exc()
    else:
        if args.device == 'cpu':
            _SHARED_PIPE = DiffusionPipeline.from_pretrained(args.model).to(args.device)
        else:
            _SHARED_PIPE = DiffusionPipeline.from_pretrained(args.model, device_map='balanced')
            try:
                cache_dit.enable_cache(_SHARED_PIPE)
            except Exception as e:
                print(f"[INFO] cache_dit not enabled (model not DiT?): {e}")

    assert isinstance(_SHARED_PIPE, DiffusionPipeline), "pipeline init error, don't forget to assign pipeline to pipe var"

