from api import app
from api.image_generation import ImageGenerationAPI
from core import init_pipeline
from utils import parse_args
from config import init_config
from metrics import set_server_info
import logging
import sys
import uvicorn

def start_service():
    args = parse_args()
    init_config(args)
    
    # Set server info for metrics
    set_server_info(args.model, args.device, args.host, args.port)
    
    level = logging._nameToLevel[args.logging_level]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout
    )
    logging.info(f"Loading model {args.model} on device {args.device}...")
    init_pipeline(args)
    logging.info(f"model {args.model} loaded")
    image_api = ImageGenerationAPI()
    app.include_router(image_api.get_router())

    uvicorn.run(app, host=args.host, port=args.port)
