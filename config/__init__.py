from typing import NamedTuple

class Config(NamedTuple):
    model: str
    host: str
    port: int
    device: str
    served_model_name: str

_config: Config = None

def init_config(args):
    global _config
    _config = Config(
        model=args.model,
        host=args.host,
        port=args.port,
        device=args.device,
        served_model_name=args.served_model_name
    )

def get_config():
    global _config
    if _config is None:
        raise Exception("Config not initialized")
    return _config
