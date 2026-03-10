from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import text, image


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    yield
    print("Shutting down...")


app = FastAPI(
    title="DataScience HF MLOps API",
    description="API for text and image classification",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(text.router, prefix="/text", tags=["Text"])
app.include_router(image.router, prefix="/image", tags=["Image"])


@app.get("/")
def root():
    return {"message": "Welcome to DataScience HF MLOps API"}


@app.get("/health")
def health():
    return {"status": "healthy"}
