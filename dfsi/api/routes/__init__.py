"""HTTP routes, one module per area of the dashboard."""

from . import decisions, rehearsal, replays, status

ROUTERS = (status.router, replays.router, rehearsal.router, decisions.router)

__all__ = ["ROUTERS"]
