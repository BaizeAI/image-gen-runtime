import argparse
import torch

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
    parser.add_argument(
        "--served-model-name",
        type=str,
        default='',
        help="The model name that this service is serving, used to validate requests."
    )
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.served_model_name:
        args.served_model_name = args.model

    return args
