from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"]
)

PROCESS_RSS_MB = Gauge(
    "Process_resident_memory_mb",
    "Resident memory size (RSS) in MB"
)

PROCESS_CPU_PERCENT = Gauge(
    "process_cpu_percent",
    "Process CPU percent (psutil)",
)
