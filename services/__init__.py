from api import app
from api.image_generation import ImageGenerationAPI
from core import init_pipeline
from utils import parse_args
from config import init_config
import logging
import sys

def start_service():
    args = parse_args()
    init_config(args)
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
    app.register_blueprint(image_api.get_blueprint())

    app.run(host=args.host, port=args.port, threaded=True)
