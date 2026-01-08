from fastapi import FastAPI
from src.api.future_price import router as future_router

from src.monitoring.middleware import MetricsMiddleware
from src.monitoring.routes import router as metrics_router

app = FastAPI(
    title="Stock prediction API",
    description="predicts stock price"
)

app.add_middleware(MetricsMiddleware)
app.include_router(metrics_router)


@app.get("/")
async def root():
    return {"message": "API is running!"}

app.include_router(future_router)
