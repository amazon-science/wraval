"""WRAVAL Webapp - FastAPI application."""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="WRAVAL Webapp")

from wraval.webapp.routers.prompts import router as prompt_router
from wraval.webapp.routers.inference import router as inference_router
from wraval.webapp.routers.judge import router as judge_router
from wraval.webapp.routers.data import router as data_router
from wraval.webapp.routers.config import router as config_router
from wraval.webapp.routers.jobs import router as jobs_router

app.include_router(prompt_router, prefix="/api/prompts")
app.include_router(inference_router, prefix="/api/inference")
app.include_router(judge_router, prefix="/api/judge")
app.include_router(data_router, prefix="/api/data")
app.include_router(config_router, prefix="/api/config")
app.include_router(jobs_router, prefix="/api/jobs")

# Serve frontend static files — static dir created in task 1.3
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True))
