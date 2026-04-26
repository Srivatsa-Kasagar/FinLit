"""Verify the MCP integration raises a helpful ImportError when mcp is missing."""
import sys
from unittest.mock import patch

import pytest


def test_missing_mcp_extra_raises_helpful_importerror():
    # Pretend the `mcp` package is not installed.
    blocked = {name: None for name in list(sys.modules) if name == "mcp" or name.startswith("mcp.")}
    blocked["mcp"] = None

    # Force re-import of finlit.integrations.mcp from scratch.
    for mod in list(sys.modules):
        if mod.startswith("finlit.integrations.mcp"):
            del sys.modules[mod]

    with patch.dict(sys.modules, blocked):
        with pytest.raises(ImportError, match=r"finlit\[mcp\] extras not installed"):
            import finlit.integrations.mcp  # noqa: F401
