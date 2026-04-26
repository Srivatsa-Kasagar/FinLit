"""MCP integration for FinLit. Install with: pip install finlit[mcp]."""
try:
    from finlit.integrations.mcp.server import serve
except ImportError as exc:  # pragma: no cover - exercised via sys.modules patching
    raise ImportError(
        "finlit[mcp] extras not installed. "
        "Run: pip install finlit[mcp]"
    ) from exc

__all__ = ["serve"]
