from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
from pathlib import Path

from .routers.analysis import router as analysis_router

app = FastAPI(
    title="BoneAgeTW2",
    description="Estimación automática de maduración ósea — Método Tanner-Whitehouse 2 (20 huesos)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router)

_REF_DIR = Path(__file__).parent.parent / "ml" / "reference_data"


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/reference/tw2-tables")
def tw2_tables():
    return json.loads((_REF_DIR / "tw2_tables.json").read_text())


@app.get("/reference/gaussian-params")
def gaussian_params():
    return json.loads((_REF_DIR / "gaussian_params.json").read_text())
