"""DFSI API — the app factory.

Endpoints (OpenAPI at /docs)
----------------------------
GET  /api/health              service, model and agent status
GET  /api/replays             the scenario catalogue (the trigger menu)
GET  /api/model-card          what the model is, how it scored, its limitations
POST /api/replay/stream       SSE: stream a scenario past the forecaster
POST /api/drill               run the rehearsal room on a replay
POST /api/explain             run the two Explainer agents + validator
POST /api/decision            record a human decision (simulated, audited)
GET  /api/decisions           read the audit log, newest first
GET  /api/policies            the simulated policy store and its history
POST /api/policies/reset      restore the store to raw.json
GET  /api/risk-thresholds     model default and per-range risk thresholds

Everything is read-only with respect to the DDoS control plane. The only
writes are to DFSI's own audit log and its in-memory simulated policy store.

Layout: routes (`routes/`) translate HTTP to calls on `DfsiServices`
(`services.py`), which holds the collaborators and every use case. Tests build
an app with `create_app(build_services(forecaster=fake, ...))`.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .errors import DfsiError
from .routes import ROUTERS
from .services import DfsiServices

#: The dashboard is served by Vite in development, so its origin differs from
#: the API's. Restricted to the local dev origins rather than "*" — a wildcard
#: with credentials enabled is how a dashboard becomes an open proxy.
_DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


def create_app(services: DfsiServices | None = None) -> FastAPI:
    """Build the API. Without `services`, production wiring is built on first request."""
    app = FastAPI(
        title="DFSI — DDoS Future-State Intelligence",
        description=(
            "Predictive lookout beside netsec-ddos-shared-state. Read-only and "
            "simulation-only: ML makes the numbers, AI explains them, humans decide."
        ),
        version="0.1.0",
    )
    app.state.services = services
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(DfsiError)
    async def _domain_error(_: Request, exc: DfsiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    for router in ROUTERS:
        app.include_router(router)
    return app


def _cors_origins() -> list[str]:
    raw = os.getenv("DFSI_CORS_ORIGINS", _DEFAULT_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


#: The ASGI app uvicorn serves (`uvicorn dfsi.api.app:app`).
app = create_app()
