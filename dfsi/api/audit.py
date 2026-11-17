"""The decision audit log: append-only JSON Lines, safe under concurrent writes.

FastAPI runs synchronous endpoints on a thread pool, so two operators approving
at once would otherwise race on the same file handle and could interleave
partial lines. Every append and every read takes the same lock, and a read of
the newest records seeks from the end of the file rather than loading it all.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger("dfsi.api")

#: Bytes read per step when scanning backwards for the newest records.
_BLOCK = 8192


class AuditLog:
    """Append-only, newest-first readable log of simulated human decisions."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        """Write one record as a single line, flushed and fsynced before returning."""
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # A crash mid-write can leave a last line without its newline;
            # start on a fresh line so this record is never glued onto it.
            if _ends_mid_line(self.path):
                line = "\n" + line
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    def tail(self, limit: int) -> list[dict[str, Any]]:
        """The newest `limit` records, newest first. Reads only the end of the file."""
        if limit <= 0:
            return []
        with self._lock:
            if not self.path.exists():
                return []
            records: list[dict[str, Any]] = []
            for line in _lines_backwards(self.path):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # One torn line must not take the whole log offline.
                    log.warning("skipping an unreadable audit record in %s", self.path)
                    continue
                if len(records) == limit:
                    break
            return records


def _ends_mid_line(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) != b"\n"


def _lines_backwards(path: Path) -> Iterator[str]:
    """Yield the file's lines last-first, reading fixed-size blocks from the end."""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        remainder = b""
        while position > 0:
            step = min(_BLOCK, position)
            position -= step
            handle.seek(position)
            chunk = handle.read(step) + remainder
            lines = chunk.split(b"\n")
            # The first piece may be the tail of a line that starts in an
            # earlier block; keep it until that block has been read.
            remainder = lines.pop(0)
            for raw in reversed(lines):
                yield raw.decode("utf-8")
        if remainder:
            yield remainder.decode("utf-8")
