from fastapi import FastAPI
from src.api.future_price import router as future_router

app = FastAPI(
    title="Stock prediction API",
    description="predicts stock price"
)


@app.get("/")
async def root():
    return {"message": "API is running!"}

app.include_router(future_router)
