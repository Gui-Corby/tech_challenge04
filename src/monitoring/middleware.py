import time
import psutil
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    PROCESS_RSS_MB,
    PROCESS_CPU_PERCENT,
)

_process = psutil.Process()


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        status_code = 500

        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - start

            path = request.url.path

            REQUEST_LATENCY.labels(request.method, path).observe(elapsed)
            REQUEST_COUNT.labels(request.method, path, str(status_code)).inc()

            rss_mb = _process.memory_info().rss / (1024 * 1024)
            PROCESS_RSS_MB.set(rss_mb)

            cpu_pct = _process.cpu_percent(interval=None)
            PROCESS_CPU_PERCENT.set(cpu_pct)
