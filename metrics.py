"""
Prometheus metrics for image generation service.
"""
import os
import time
from typing import Callable, Any
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, Info

# Get metrics prefix from environment variable, default to 'image_gen:'
METRICS_PREFIX = os.getenv('METRICS_PREFIX', 'image_gen:')

# Global model name for all metrics
MODEL_NAME = None

def _metric_name(name: str) -> str:
    """Add prefix to metric name."""
    return f"{METRICS_PREFIX}{name}"

def set_model_name(model_name: str):
    """Set the global model name for metrics."""
    global MODEL_NAME
    MODEL_NAME = model_name


# Image generation metrics
REQUEST_COUNT = Counter(
    _metric_name('image_generation_requests_total'), 
    'Total image generation requests',
    ['model_name', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    _metric_name('image_generation_request_duration_seconds'),
    'Image generation request duration',
    ['model_name', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    _metric_name('image_generation_active_requests'),
    'Number of active image generation requests',
    ['model_name']
)

IMAGES_GENERATED = Counter(
    _metric_name('images_generated_total'),
    'Total number of images generated',
    ['model_name']
)

# Pipeline metrics  
PIPELINE_LOAD_DURATION = Histogram(
    _metric_name('pipeline_load_duration_seconds'),
    'Time taken to load the pipeline model',
    ['model_name'],
    buckets=[5, 10, 30, 60, 120, 300, 600, float('inf')]  # 5s to 10min+ for model loading
)

INFERENCE_DURATION = Histogram(
    _metric_name('inference_duration_seconds'), 
    'Time taken for image inference',
    ['model_name', 'quality', 'size']
)

PIPELINE_MEMORY_USAGE = Gauge(
    _metric_name('pipeline_memory_usage_bytes'),
    'Memory usage by the pipeline',
    ['model_name']
)

# Health check metrics - removed as per feedback

# Server info
SERVER_INFO = Info(
    _metric_name('server_info'),
    'Server information',
    ['model_name']
)

# Request size metrics
REQUEST_SIZE_HISTOGRAM = Histogram(
    _metric_name('request_image_size_pixels'),
    'Histogram of requested image sizes',
    ['model_name', 'size']  # Put size into label instead of using buckets
)

INFERENCE_STEPS_HISTOGRAM = Histogram(
    _metric_name('inference_steps_count'),
    'Histogram of inference steps used',
    ['model_name'],
    buckets=[10, 20, 30, 40, 50, 75, 100]
)


def track_request_metrics(endpoint: str):
    """Decorator to track request metrics for an endpoint."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            ACTIVE_REQUESTS.labels(model_name=MODEL_NAME).inc()
            
            try:
                result = await func(*args, **kwargs)
                REQUEST_COUNT.labels(model_name=MODEL_NAME, endpoint=endpoint, status='success').inc()
                return result
            except Exception as e:
                REQUEST_COUNT.labels(model_name=MODEL_NAME, endpoint=endpoint, status='error').inc()
                raise
            finally:
                REQUEST_DURATION.labels(model_name=MODEL_NAME, endpoint=endpoint).observe(time.time() - start_time)
                ACTIVE_REQUESTS.labels(model_name=MODEL_NAME).dec()
                
        return wrapper
    return decorator


def track_pipeline_load(func: Callable) -> Callable:
    """Decorator to track pipeline loading time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            PIPELINE_LOAD_DURATION.labels(model_name=MODEL_NAME).observe(time.time() - start_time)
            return result
        except Exception:
            raise
    return wrapper


def track_inference_metrics(func: Callable) -> Callable:
    """Decorator to track inference metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract request object to get quality and size info
        req = args[1] if len(args) > 1 else kwargs.get('req')
        if req:
            # Use size in label format (e.g. "512x512")
            size_label = f"{req.width}x{req.height}"
            REQUEST_SIZE_HISTOGRAM.labels(model_name=MODEL_NAME, size=size_label).observe(req.width * req.height)
            INFERENCE_STEPS_HISTOGRAM.labels(model_name=MODEL_NAME).observe(req.num_inference_steps)
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            quality = req.quality if req else 'unknown'
            size = req.size if req else 'unknown'
            INFERENCE_DURATION.labels(model_name=MODEL_NAME, quality=quality, size=size).observe(time.time() - start_time)
            
            # Count generated images
            if hasattr(result, 'images') and result.images:
                IMAGES_GENERATED.labels(model_name=MODEL_NAME).inc(len(result.images))
            
            return result
        except Exception:
            raise
            
    return wrapper


def set_server_info(model: str, device: str, host: str, port: int):
    """Set server information metrics."""
    # Set the global model name
    set_model_name(model)
    
    SERVER_INFO.labels(model_name=MODEL_NAME).info({
        'model': model,
        'device': device, 
        'host': host,
        'port': str(port),
        'version': '1.0.0'
    })