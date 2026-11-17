"""The decision audit log: ordering, bounded reads, and concurrent appends."""

from __future__ import annotations

import json
import threading

from dfsi.api import audit
from dfsi.api.audit import AuditLog


def test_a_missing_log_reads_as_empty(tmp_path):
    assert AuditLog(tmp_path / "none.jsonl").tail(10) == []


def test_tail_is_newest_first_and_bounded(tmp_path):
    log = AuditLog(tmp_path / "d" / "decisions.jsonl")  # parent created on first write
    for i in range(7):
        log.append({"seq": i})
    assert [r["seq"] for r in log.tail(3)] == [6, 5, 4]
    assert [r["seq"] for r in log.tail(100)] == list(range(6, -1, -1))
    assert log.tail(0) == []


def test_tail_reads_across_block_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "_BLOCK", 16)  # force many tiny blocks
    log = AuditLog(tmp_path / "decisions.jsonl")
    records = [{"seq": i, "note": "x" * (i * 7)} for i in range(25)]
    for record in records:
        log.append(record)
    assert log.tail(25) == records[::-1]


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "decisions.jsonl"
    path.write_text('{"seq": 0}\n\n{"seq": 1}\n\n')
    assert [r["seq"] for r in AuditLog(path).tail(5)] == [1, 0]


def test_concurrent_appends_never_interleave(tmp_path):
    log = AuditLog(tmp_path / "decisions.jsonl")
    writers, per_writer = 8, 50
    payload = "y" * 2_000  # long enough that an unlocked write could split

    def write(writer: int) -> None:
        for n in range(per_writer):
            log.append({"writer": writer, "n": n, "payload": payload})

    threads = [threading.Thread(target=write, args=(w,)) for w in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = (tmp_path / "decisions.jsonl").read_text().splitlines()
    assert len(lines) == writers * per_writer
    parsed = [json.loads(line) for line in lines]  # every line is whole JSON
    for w in range(writers):
        assert [r["n"] for r in parsed if r["writer"] == w] == list(range(per_writer))


def test_a_torn_line_is_skipped_and_never_swallows_the_next_record(tmp_path):
    path = tmp_path / "decisions.jsonl"
    path.write_text('{"seq": 0}\n{"seq": 1, "note": "cut off by a cra')  # no newline
    log = AuditLog(path)
    assert [r["seq"] for r in log.tail(5)] == [0]  # readable, not a 500
    log.append({"seq": 2})
    assert [r["seq"] for r in log.tail(5)] == [2, 0]  # on its own line, not glued on
