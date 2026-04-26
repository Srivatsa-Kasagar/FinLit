"""FinLit MCP server - FastMCP app + tool registrations + serve() entry point.

The module exposes:

  - build_app(...)  - build a FastMCP app with the given server-startup config.
                       Pure construction; no I/O. Used by tests.
  - serve(...)      - build_app + run stdio. The CLI and __main__ launchers
                       both call this.
"""
from __future__ import annotations

from typing import Literal

from mcp.server.fastmcp import FastMCP

from finlit.integrations._schema_resolver import _DOTTED_TO_ATTR, _resolve_schema

PIIMode = Literal["redact", "raw"]


def build_app(
    *,
    extractor: str,
    vision_extractor: str | None,
    review_threshold: float,
    pii_mode: PIIMode,
) -> FastMCP:
    """Construct a FastMCP app with the given server-startup configuration."""
    app = FastMCP("finlit")

    # Server-startup config is captured in the closures below.
    server_default_redact = pii_mode == "redact"

    @app.tool()
    def list_schemas() -> list[dict]:
        """List all built-in FinLit schemas with field counts and required fields."""
        out = []
        for dotted_key in sorted(_DOTTED_TO_ATTR):
            schema = _resolve_schema(dotted_key)
            out.append({
                "key": dotted_key,
                "name": schema.document_type or schema.name,
                "version": schema.version,
                "field_count": len(schema.fields),
                "required_fields": [f.name for f in schema.fields if f.required],
                "description": schema.description,
            })
        return out

    # Stash config on the app for downstream tools added in later tasks.
    app._finlit_extractor = extractor                # type: ignore[attr-defined]
    app._finlit_vision = vision_extractor            # type: ignore[attr-defined]
    app._finlit_threshold = review_threshold         # type: ignore[attr-defined]
    app._finlit_default_redact = server_default_redact  # type: ignore[attr-defined]

    return app


def serve(
    *,
    extractor: str = "claude",
    vision_extractor: str | None = None,
    review_threshold: float = 0.85,
    pii_mode: PIIMode = "redact",
) -> None:
    """Build the app and run it over stdio. Blocks until the host disconnects."""
    app = build_app(
        extractor=extractor,
        vision_extractor=vision_extractor,
        review_threshold=review_threshold,
        pii_mode=pii_mode,
    )
    app.run()  # FastMCP defaults to stdio transport.
