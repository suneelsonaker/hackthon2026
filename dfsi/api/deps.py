"""FastAPI dependency: the `DfsiServices` the running app was built with."""

from __future__ import annotations

import threading
from typing import Annotated

from fastapi import Depends, Request

from .services import DfsiServices, build_services

_build_lock = threading.Lock()


def get_services(request: Request) -> DfsiServices:
    """The app's services, built on first use when the app was created without any.

    Building lazily (rather than at import) keeps `import dfsi.api.app` cheap and
    side-effect free, and still works for a `TestClient` used without its
    context manager, where the lifespan never runs.
    """
    state = request.app.state
    services = getattr(state, "services", None)
    if services is None:
        with _build_lock:
            services = getattr(state, "services", None)
            if services is None:
                services = state.services = build_services()
    return services


Services = Annotated[DfsiServices, Depends(get_services)]
