"""Tests for finlit.audit.audit_log.AuditLog."""
import json

import pytest

from finlit.audit.audit_log import AuditLog


def test_log_appends_event_with_timestamp():
    log = AuditLog(run_id="r1")
    log.log("parse_start", file="x.pdf")
    events = log.to_dict()
    assert len(events) == 1
    assert events[0]["event"] == "parse_start"
    assert events[0]["file"] == "x.pdf"
    assert "ts" in events[0]


def test_multiple_events_preserved_in_order():
    log = AuditLog(run_id="r1")
    log.log("a")
    log.log("b")
    log.log("c")
    names = [e["event"] for e in log.to_dict()]
    assert names == ["a", "b", "c"]


def test_log_raises_after_finalize():
    log = AuditLog(run_id="r1")
    log.log("ok")
    log.finalize()
    with pytest.raises(RuntimeError):
        log.log("should_fail")


def test_to_json_returns_valid_json_string():
    log = AuditLog(run_id="r1")
    log.log("hello", data=42)
    parsed = json.loads(log.to_json())
    assert parsed[0]["event"] == "hello"
    assert parsed[0]["data"] == 42
