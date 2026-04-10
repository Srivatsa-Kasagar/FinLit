"""
Structured, append-only audit log for every pipeline run.
Each event is a timestamped dict. The log is immutable once finalized.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AuditLog:
    run_id: str
    events: list[dict] = field(default_factory=list)
    _finalized: bool = False

    def log(self, event: str, **kwargs: Any) -> None:
        if self._finalized:
            raise RuntimeError("Audit log is finalized and cannot be modified.")
        self.events.append({"event": event, "ts": _now(), **kwargs})

    def finalize(self) -> None:
        self._finalized = True

    def to_dict(self) -> list[dict]:
        return list(self.events)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
