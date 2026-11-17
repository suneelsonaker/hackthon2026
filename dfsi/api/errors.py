"""Domain errors raised by the service layer, mapped to HTTP status codes once.

The services never import FastAPI. They raise these, and `app.py` registers a
single handler that turns each into a JSON error with its status code, so no
route has to repeat the translation.
"""

from __future__ import annotations


class DfsiError(Exception):
    """Base class: carries the HTTP status code the API should answer with."""

    status_code = 500

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class UnknownReplay(DfsiError):
    """The replay id is not in the catalogue."""

    status_code = 404

    def __init__(self, replay_id: str) -> None:
        super().__init__(f"unknown replay {replay_id!r}")
        self.replay_id = replay_id


class InvalidEpoch(DfsiError):
    """The requested decision epoch is not a window of the replay."""

    status_code = 400


class InvalidRequest(DfsiError):
    """The request is well-formed but asks for something that cannot be computed."""

    status_code = 422


class NotBuilt(DfsiError):
    """A generated artefact (Memory layer, training report) is missing."""

    status_code = 503
