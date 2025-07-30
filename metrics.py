"""
Prometheus metrics for image generation service.
"""
import time
from typing import Callable, Any
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, Info


# Image generation metrics
REQUEST_COUNT = Counter(
    'image_generation_requests_total', 
    'Total image generation requests',
    ['endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'image_generation_request_duration_seconds',
    'Image generation request duration',
    ['endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'image_generation_active_requests',
    'Number of active image generation requests'
)

IMAGES_GENERATED = Counter(
    'images_generated_total',
    'Total number of images generated'
)

# Pipeline metrics  
PIPELINE_LOAD_DURATION = Histogram(
    'pipeline_load_duration_seconds',
    'Time taken to load the pipeline model'
)

INFERENCE_DURATION = Histogram(
    'inference_duration_seconds', 
    'Time taken for image inference',
    ['quality', 'size']
)

PIPELINE_MEMORY_USAGE = Gauge(
    'pipeline_memory_usage_bytes',
    'Memory usage by the pipeline'
)

# Health check metrics
HEALTH_CHECK_COUNT = Counter(
    'health_checks_total',
    'Total health check requests', 
    ['status']
)

# Server info
SERVER_INFO = Info(
    'server_info',
    'Server information'
)

# Request size metrics
REQUEST_SIZE_HISTOGRAM = Histogram(
    'request_image_size_pixels',
    'Histogram of requested image sizes',
    buckets=[256*256, 512*512, 768*768, 1024*1024, 1536*1536, 2048*2048]
)

INFERENCE_STEPS_HISTOGRAM = Histogram(
    'inference_steps_count',
    'Histogram of inference steps used',
    buckets=[10, 20, 30, 40, 50, 75, 100]
)


def track_request_metrics(endpoint: str):
    """Decorator to track request metrics for an endpoint."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            ACTIVE_REQUESTS.inc()
            
            try:
                result = await func(*args, **kwargs)
                REQUEST_COUNT.labels(endpoint=endpoint, status='success').inc()
                return result
            except Exception as e:
                REQUEST_COUNT.labels(endpoint=endpoint, status='error').inc()
                raise
            finally:
                REQUEST_DURATION.labels(endpoint=endpoint).observe(time.time() - start_time)
                ACTIVE_REQUESTS.dec()
                
        return wrapper
    return decorator


def track_pipeline_load(func: Callable) -> Callable:
    """Decorator to track pipeline loading time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            PIPELINE_LOAD_DURATION.observe(time.time() - start_time)
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
            REQUEST_SIZE_HISTOGRAM.observe(req.width * req.height)
            INFERENCE_STEPS_HISTOGRAM.observe(req.num_inference_steps)
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            quality = req.quality if req else 'unknown'
            size = req.size if req else 'unknown'
            INFERENCE_DURATION.labels(quality=quality, size=size).observe(time.time() - start_time)
            
            # Count generated images
            if hasattr(result, 'images') and result.images:
                IMAGES_GENERATED.inc(len(result.images))
            
            return result
        except Exception:
            raise
            
    return wrapper


def set_server_info(model: str, device: str, host: str, port: int):
    """Set server information metrics."""
    SERVER_INFO.info({
        'model': model,
        'device': device, 
        'host': host,
        'port': str(port),
        'version': '1.0.0'
    })