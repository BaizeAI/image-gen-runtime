# Prometheus Metrics

This document describes the Prometheus metrics exposed by the image generation service.

## Configuration

### Metrics Prefix

All metrics are prefixed with a configurable namespace. By default, metrics use the `image_gen:` prefix.

You can customize the prefix using the `METRICS_PREFIX` environment variable:

```bash
export METRICS_PREFIX="my_app:"
# or
METRICS_PREFIX="custom_prefix:" python main.py
```

**Default**: `image_gen:`

## Metrics Endpoint

The service exposes metrics at `/metrics` endpoint in standard Prometheus format.

```bash
curl http://localhost:8000/metrics
```

## Available Metrics

### Request Metrics

#### `image_gen:image_generation_requests_total`
- **Type**: Counter
- **Description**: Total number of image generation requests
- **Labels**:
  - `endpoint`: The API endpoint (e.g., "image_generation", "health_check")
  - `status`: Request outcome ("success" or "error")

#### `image_gen:image_generation_request_duration_seconds`
- **Type**: Histogram
- **Description**: Duration of image generation requests in seconds
- **Labels**:
  - `endpoint`: The API endpoint

#### `image_gen:image_generation_active_requests`
- **Type**: Gauge
- **Description**: Number of currently active image generation requests

### Image Generation Metrics

#### `image_gen:images_generated_total`
- **Type**: Counter
- **Description**: Total number of images successfully generated

#### `image_gen:inference_duration_seconds`
- **Type**: Histogram
- **Description**: Time taken for image inference in seconds
- **Labels**:
  - `quality`: Image quality setting (e.g., "hd")
  - `size`: Image dimensions (e.g., "512x512", "1024x768")

#### `image_gen:request_image_size_pixels`
- **Type**: Histogram
- **Description**: Distribution of requested image sizes in pixels
- **Buckets**: 256², 512², 768², 1024², 1536², 2048²

#### `image_gen:inference_steps_count`
- **Type**: Histogram  
- **Description**: Distribution of inference steps used in generation
- **Buckets**: 10, 20, 30, 40, 50, 75, 100

### Pipeline Metrics

#### `image_gen:pipeline_load_duration_seconds`
- **Type**: Histogram
- **Description**: Time taken to load the diffusion pipeline model

#### `image_gen:pipeline_memory_usage_bytes`
- **Type**: Gauge
- **Description**: Current memory usage by the pipeline (when available)

### Health Check Metrics

#### `image_gen:health_checks_total`
- **Type**: Counter
- **Description**: Total number of health check requests
- **Labels**:
  - `status`: Health check result ("success" or "failure")

### Server Information

#### `image_gen:server_info`
- **Type**: Info
- **Description**: Static information about the server
- **Labels**:
  - `model`: The loaded model name
  - `device`: Device used for inference (cpu/cuda)
  - `host`: Server host address
  - `port`: Server port
  - `version`: Service version

## Example Queries

### Request Rate
```promql
rate(image_gen:image_generation_requests_total[5m])
```

### Error Rate
```promql
rate(image_gen:image_generation_requests_total{status="error"}[5m]) / rate(image_gen:image_generation_requests_total[5m])
```

### Average Request Duration
```promql
rate(image_gen:image_generation_request_duration_seconds_sum[5m]) / rate(image_gen:image_generation_request_duration_seconds_count[5m])
```

### 95th Percentile Response Time
```promql
histogram_quantile(0.95, rate(image_gen:image_generation_request_duration_seconds_bucket[5m]))
```

### Images Generated Per Second
```promql
rate(image_gen:images_generated_total[5m])
```

### Average Inference Time by Quality
```promql
rate(image_gen:inference_duration_seconds_sum[5m]) / rate(image_gen:inference_duration_seconds_count[5m]) by (quality)
```

### Active Requests
```promql
image_gen:image_generation_active_requests
```

### Pipeline Load Time
```promql
image_gen:pipeline_load_duration_seconds
```

## Alerting Examples

### High Error Rate
```yaml
- alert: HighImageGenerationErrorRate
  expr: rate(image_gen:image_generation_requests_total{status="error"}[5m]) / rate(image_gen:image_generation_requests_total[5m]) > 0.1
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: High error rate in image generation
    description: "Error rate is {{ $value | humanizePercentage }} over the last 5 minutes"
```

### High Response Time
```yaml
- alert: HighImageGenerationLatency
  expr: histogram_quantile(0.95, rate(image_gen:image_generation_request_duration_seconds_bucket[5m])) > 30
  for: 5m
  labels:
    severity: warning  
  annotations:
    summary: High latency in image generation
    description: "95th percentile latency is {{ $value }}s over the last 5 minutes"
```

### Service Health Check Failures
```yaml
- alert: ImageGenerationServiceUnhealthy
  expr: rate(image_gen:health_checks_total{status="failure"}[5m]) > 0
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: Image generation service failing health checks
    description: "Service has been failing health checks for more than 2 minutes"
```

## Grafana Dashboard

You can create a Grafana dashboard to visualize these metrics. Key panels to include:

1. **Request Rate**: `rate(image_gen:image_generation_requests_total[5m])`
2. **Error Rate**: Error rate calculation as shown above
3. **Response Time Distribution**: Histogram of request durations
4. **Active Requests**: Current active request gauge
5. **Images Generated**: Rate of image generation
6. **Pipeline Performance**: Inference times by quality/size
7. **Server Info**: Display model, device, and version information

## Implementation Details

The metrics are implemented using the `prometheus_client` Python library and are automatically instrumented through decorators in the codebase:

- `@track_request_metrics()` - Applied to API endpoints
- `@track_pipeline_load` - Applied to pipeline initialization  
- `@track_inference_metrics` - Applied to inference operations

This approach ensures minimal code changes while providing comprehensive monitoring coverage.