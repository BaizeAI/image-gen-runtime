from api import app
from api.image_generation import ImageGenerationAPI
from core import init_pipeline
from utils import parse_args
from config import init_config

def start_service():
    args = parse_args()
    init_config(args)
    print(f"Loading model {args.model} on device {args.device}...")
    init_pipeline(args)
    print("Done.")
    image_api = ImageGenerationAPI()
    app.register_blueprint(image_api.get_blueprint())

    app.run(host=args.host, port=args.port, threaded=True)
