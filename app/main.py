import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

# logging setup — catches every request and any errors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),  # console
        logging.FileHandler("app.log"),  # file
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="predicts if a customer is gonna leave using ML",
    version="1.0.0",
)

# allow requests from anywhere (needed for frontend / postman)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# log every single request with how long it took
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)

    logger.info(f"{request.method} {request.url.path} | {response.status_code} | {duration}ms")
    return response


app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is running",
        "docs": "/docs",
        "health": "/api/v1/health",
        "predict": "/api/v1/predict",
    }
