# backend/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes_ct import router as ct_router
from app.core.config import get_settings
from app.models.ct_model_loader import load_ct_model


settings = get_settings()

app = FastAPI(title=settings.PROJECT_NAME)

# allow frontend origin for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    # preload model into memory
    load_ct_model()


app.include_router(ct_router, prefix="/api/v1", tags=["ct"])
