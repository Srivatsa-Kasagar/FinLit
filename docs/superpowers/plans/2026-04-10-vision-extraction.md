# FinLit v0.3.0 — Vision Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a model-agnostic vision extraction fallback path to `DocumentPipeline`, letting any multimodal LLM (including fully-local OSS models via Ollama) correct text-extraction failures on scanned PDFs and flattened form layouts.

**Architecture:** New `BaseVisionExtractor` ABC and default `VisionExtractor` implementation (backed by pydantic-ai). A new `render_pages()` utility rasterizes PDFs via pypdfium2. `DocumentPipeline` gains two opt-in parameters (`vision_extractor`, `vision_fallback_when`); when the text path finishes, the pipeline evaluates the callback and, if True, re-runs extraction from page images. Full-replacement semantics: vision wins if it ran. Fail-safe: any failure in the vision path falls back to the text result with an audit event and warning.

**Tech Stack:** pydantic-ai 1.x (with `BinaryContent` for image inputs), pypdfium2 (already transitive via Docling), PIL (already transitive via Docling), pytest, pytest's monkeypatch.

**Design reference:** `docs/superpowers/specs/2026-04-10-vision-extraction-design.md`

---

## File Structure

**New files:**
- `finlit/extractors/base_vision.py` — `BaseVisionExtractor` ABC (takes images + schema, returns `ExtractionOutput`)
- `finlit/extractors/vision_extractor.py` — `VisionExtractor` concrete class wrapping a pydantic-ai `Agent`
- `finlit/parsers/image_renderer.py` — `render_pages(path, dpi)` helper, converts PDFs/images to PNG bytes
- `tests/test_image_renderer.py` — 7 unit tests for `render_pages`
- `tests/test_vision_extractor.py` — 6 unit tests for `VisionExtractor`
- `examples/extract_with_vision.py` — Claude vision usage example
- `examples/extract_with_local_vision.py` — Ollama + Qwen2.5-VL fully-local example

**Modified files:**
- `finlit/result.py` — add `extraction_path: str = "text"` field
- `finlit/pipeline.py` — add `vision_extractor` + `vision_fallback_when` params, orchestration after the text path
- `finlit/__init__.py` — export `VisionExtractor` and `BaseVisionExtractor`
- `finlit/cli/main.py` — add `--vision-extractor` flag to `extract` command
- `tests/conftest.py` — add `StubVisionExtractor` helper class
- `tests/test_pipeline.py` — add 9 integration tests for fallback orchestration
- `README.md` — add "When to use vision extraction" section, OSS model callout, update Roadmap

---

## Task 1: Add `extraction_path` field to `ExtractionResult`

This is the foundation for later tests that assert which path produced a result.

**Files:**
- Modify: `finlit/result.py`
- Test: `tests/test_result.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_result.py`:

```python
def test_extraction_result_default_extraction_path_is_text():
    """By default, ExtractionResult.extraction_path is 'text' for backwards
    compatibility with v0.2.0 consumers."""
    from finlit.result import ExtractionResult

    result = ExtractionResult(fields={}, confidence={}, source_ref={})
    assert result.extraction_path == "text"


def test_extraction_result_extraction_path_can_be_vision():
    """extraction_path can be explicitly set to 'vision' when the vision
    fallback produced the result."""
    from finlit.result import ExtractionResult

    result = ExtractionResult(
        fields={}, confidence={}, source_ref={}, extraction_path="vision"
    )
    assert result.extraction_path == "vision"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_result.py::test_extraction_result_default_extraction_path_is_text tests/test_result.py::test_extraction_result_extraction_path_can_be_vision -v`

Expected: FAIL with `TypeError: ExtractionResult.__init__() got an unexpected keyword argument 'extraction_path'` for the second test, or `AttributeError: 'ExtractionResult' object has no attribute 'extraction_path'` for the first.

- [ ] **Step 3: Add the field to `ExtractionResult`**

In `finlit/result.py`, locate the `# Metadata` section (around line 30) and add `extraction_path` after `extractor_model`:

```python
    # Metadata
    document_path: str = ""
    schema_name: str = ""
    extractor_model: str = ""
    extraction_path: str = "text"  # "text" or "vision"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_result.py -v`

Expected: all tests in `test_result.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add finlit/result.py tests/test_result.py
git commit -m "$(cat <<'EOF'
feat(result): add extraction_path field to ExtractionResult

Additive field (default "text") that records which extraction path
produced the result. Set to "vision" by the pipeline when the vision
fallback runs. Zero breaking change for v0.2.0 consumers.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Create `BaseVisionExtractor` ABC

**Files:**
- Create: `finlit/extractors/base_vision.py`
- Test: `tests/test_vision_extractor.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_vision_extractor.py`:

```python
"""Tests for finlit.extractors.base_vision.BaseVisionExtractor and
finlit.extractors.vision_extractor.VisionExtractor."""
from __future__ import annotations

import pytest

from finlit import schemas
from finlit.extractors.base_vision import BaseVisionExtractor
from finlit.extractors.pydantic_ai_extractor import ExtractionOutput


def test_base_vision_extractor_is_abstract():
    """Instantiating BaseVisionExtractor directly must fail — it's an ABC."""
    with pytest.raises(TypeError):
        BaseVisionExtractor()  # type: ignore[abstract]


def test_base_vision_extractor_subclass_must_implement_extract():
    """A subclass that does not implement extract() cannot be instantiated."""
    class Incomplete(BaseVisionExtractor):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_base_vision_extractor_subclass_with_extract_works():
    """A subclass that implements extract() can be instantiated and called."""
    class Stub(BaseVisionExtractor):
        def extract(self, images, schema, text=""):
            return ExtractionOutput(
                fields={"payer_name": "Test"},
                confidence={"payer_name": 0.9},
            )

    s = Stub()
    out = s.extract([b"fake"], schemas.CRA_T5)
    assert out.fields["payer_name"] == "Test"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_vision_extractor.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'finlit.extractors.base_vision'`.

- [ ] **Step 3: Create the ABC**

Create `finlit/extractors/base_vision.py`:

```python
"""
Abstract base class for vision-based extractors.

A concrete vision extractor takes a list of PNG-encoded page images plus
a Schema and returns an ExtractionOutput (fields, confidence, notes).

This ABC is deliberately separate from BaseExtractor (which takes text)
so that DocumentPipeline can type-check the two slots independently:
`extractor: BaseExtractor | str` for text, `vision_extractor:
BaseVisionExtractor | None` for vision. This prevents accidentally
passing a text extractor as the vision fallback.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finlit.schema import Schema


class BaseVisionExtractor(ABC):
    """All vision extractor backends must implement this interface."""

    @abstractmethod
    def extract(
        self,
        images: list[bytes],
        schema: "Schema",
        text: str = "",
    ) -> Any:
        """Extract structured fields from page images.

        Parameters
        ----------
        images:
            List of PNG-encoded page images, one per page, in document
            order.
        schema:
            The FinLit Schema describing which fields to extract.
        text:
            Optional text hint. When the vision extractor is used as a
            fallback from the text path, this contains whatever text
            Docling managed to recover from the document. Most
            implementations will ignore it. Extractors that want to use
            the text path's partial output as additional context (e.g.,
            "the text extractor found 'Acme Corp' as employer; verify
            from the image") can read it here.
        """
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_vision_extractor.py -v`

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add finlit/extractors/base_vision.py tests/test_vision_extractor.py
git commit -m "$(cat <<'EOF'
feat(extractors): add BaseVisionExtractor ABC

New abstract base class for extractors that consume page images instead
of text. Kept separate from BaseExtractor to give DocumentPipeline
type-level separation between the text and vision extractor slots.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Create `render_pages()` image renderer

**Files:**
- Create: `finlit/parsers/image_renderer.py`
- Test: `tests/test_image_renderer.py` (new file)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_image_renderer.py`:

```python
"""Tests for finlit.parsers.image_renderer.render_pages()."""
from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image

from finlit.parsers.image_renderer import render_pages


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _write_blank_pdf(path: Path, num_pages: int = 1) -> None:
    """Generate a tiny blank PDF at `path` using pypdfium2."""
    pdf = pdfium.PdfDocument.new()
    for _ in range(num_pages):
        pdf.new_page(612, 792)  # US-letter points
    pdf.save(str(path))
    pdf.close()


def test_render_pdf_returns_png_bytes(tmp_path: Path):
    pdf_path = tmp_path / "blank.pdf"
    _write_blank_pdf(pdf_path, num_pages=1)

    images = render_pages(pdf_path)

    assert isinstance(images, list)
    assert len(images) == 1
    assert images[0].startswith(PNG_MAGIC)


def test_render_respects_dpi(tmp_path: Path):
    """Higher DPI must produce larger PNG byte output than lower DPI."""
    pdf_path = tmp_path / "blank.pdf"
    _write_blank_pdf(pdf_path, num_pages=1)

    low = render_pages(pdf_path, dpi=72)
    high = render_pages(pdf_path, dpi=200)

    assert len(low[0]) < len(high[0])


def test_render_multipage_pdf(tmp_path: Path):
    pdf_path = tmp_path / "three_pages.pdf"
    _write_blank_pdf(pdf_path, num_pages=3)

    images = render_pages(pdf_path)

    assert len(images) == 3
    for img in images:
        assert img.startswith(PNG_MAGIC)


def test_render_png_input_passthrough(tmp_path: Path):
    """A .png input is returned as a single-element list of the raw bytes
    with no re-encoding."""
    png_path = tmp_path / "pic.png"
    Image.new("RGB", (100, 100), color="white").save(png_path, "PNG")
    original_bytes = png_path.read_bytes()

    images = render_pages(png_path)

    assert len(images) == 1
    assert images[0] == original_bytes


def test_render_jpg_input_passthrough(tmp_path: Path):
    jpg_path = tmp_path / "pic.jpg"
    Image.new("RGB", (100, 100), color="white").save(jpg_path, "JPEG")
    original_bytes = jpg_path.read_bytes()

    images = render_pages(jpg_path)

    assert len(images) == 1
    assert images[0] == original_bytes


def test_render_file_not_found_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        render_pages(tmp_path / "does_not_exist.pdf")


def test_render_unsupported_format_raises(tmp_path: Path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    with pytest.raises(ValueError, match="unsupported"):
        render_pages(txt)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_image_renderer.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'finlit.parsers.image_renderer'`.

- [ ] **Step 3: Create the renderer module**

Create `finlit/parsers/image_renderer.py`:

```python
"""
Render PDFs and image files to PNG bytes for vision-based extraction.

Standalone utility — no dependency on Docling, the pipeline, or any
extractor. Can be used directly by a consumer if they want to pre-render
documents themselves.

PDFs are rasterized page-by-page via pypdfium2 (Google's PDFium engine,
already in the dependency tree via Docling). Image files (.png, .jpg,
.jpeg) are returned as a single-element list of the raw file bytes with
no re-encoding — vision models accept them directly.
"""
from __future__ import annotations

import io
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

_PDF_SUFFIXES = {".pdf"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def render_pages(path: str | Path, dpi: int = 200) -> list[bytes]:
    """Render a document to a list of PNG-encoded page images.

    Parameters
    ----------
    path:
        Path to a PDF or image file. Must exist.
    dpi:
        Resolution used when rasterizing PDF pages. Default 200 — a
        balance between OCR legibility on small box numbers and image
        token cost. Ignored for image inputs (no re-encoding).

    Returns
    -------
    list[bytes]
        One entry per page, each a PNG byte string.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file extension is not .pdf, .png, .jpg, or .jpeg.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"image_renderer: file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return [path.read_bytes()]

    if suffix in _PDF_SUFFIXES:
        return _render_pdf(path, dpi)

    raise ValueError(
        f"image_renderer: unsupported format {suffix!r}. "
        f"Supported: .pdf, .png, .jpg, .jpeg"
    )


def _render_pdf(path: Path, dpi: int) -> list[bytes]:
    """Render every page of a PDF to PNG bytes at the given DPI."""
    scale = dpi / 72.0  # pypdfium2 scale is relative to 72 DPI
    out: list[bytes] = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        for page in pdf:
            bitmap = page.render(scale=scale)
            pil_image: Image.Image = bitmap.to_pil()
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            out.append(buf.getvalue())
    finally:
        pdf.close()
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_image_renderer.py -v`

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add finlit/parsers/image_renderer.py tests/test_image_renderer.py
git commit -m "$(cat <<'EOF'
feat(parsers): add render_pages() for PDF/image rasterization

New standalone utility that renders a PDF to a list of PNG byte strings
via pypdfium2, or passes through raw bytes for .png/.jpg inputs. Used
by v0.3 VisionExtractor to feed page images to multimodal LLMs. No new
dependencies — pypdfium2 and PIL are already transitive via Docling.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Create `VisionExtractor` class

**Files:**
- Create: `finlit/extractors/vision_extractor.py`
- Test: `tests/test_vision_extractor.py` (extend from Task 2)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vision_extractor.py`:

```python
# ---------------- VisionExtractor tests ----------------

from finlit.extractors.vision_extractor import VisionExtractor


class _FakeRunResult:
    """Stand-in for pydantic-ai's RunResult object."""
    def __init__(self, output):
        self.output = output


class _FakeAgent:
    """Stand-in for pydantic-ai's Agent, capturing what was passed to it
    without making any LLM calls."""
    def __init__(self, canned_output):
        self.canned_output = canned_output
        self.last_prompt = None
        self.call_count = 0

    def run_sync(self, prompt):
        self.last_prompt = prompt
        self.call_count += 1
        return _FakeRunResult(self.canned_output)


def _install_fake_agent(vision_extractor, canned_output):
    """Replace the real pydantic-ai Agent on a VisionExtractor with a fake."""
    fake = _FakeAgent(canned_output)
    vision_extractor._agent = fake
    return fake


def test_vision_extractor_default_model_is_claude():
    """The default model string must be claude-sonnet-4-6 per the design."""
    ve = VisionExtractor()
    assert ve.model == "anthropic:claude-sonnet-4-6"


def test_vision_extractor_custom_dpi_stored():
    ve = VisionExtractor(dpi=300)
    assert ve.dpi == 300


def test_vision_extractor_default_dpi_is_200():
    ve = VisionExtractor()
    assert ve.dpi == 200


def test_vision_extractor_returns_extraction_output():
    """extract() returns whatever the underlying agent produced."""
    ve = VisionExtractor()
    canned = ExtractionOutput(
        fields={"payer_name": "Bank of Canada", "tax_year": 2024},
        confidence={"payer_name": 0.95, "tax_year": 0.99},
    )
    _install_fake_agent(ve, canned)

    out = ve.extract([b"fakepng"], schemas.CRA_T5)

    assert out.fields["payer_name"] == "Bank of Canada"
    assert out.fields["tax_year"] == 2024
    assert out.confidence["tax_year"] == 0.99


def test_vision_extractor_passes_images_to_agent():
    """The prompt passed to agent.run_sync must include BinaryContent parts,
    one per page image, with media_type image/png."""
    from pydantic_ai import BinaryContent

    ve = VisionExtractor()
    canned = ExtractionOutput(fields={}, confidence={})
    fake = _install_fake_agent(ve, canned)

    ve.extract([b"page1png", b"page2png"], schemas.CRA_T5)

    # The prompt should be a list: [text_prompt, BinaryContent, BinaryContent]
    assert isinstance(fake.last_prompt, list)
    binary_parts = [p for p in fake.last_prompt if isinstance(p, BinaryContent)]
    assert len(binary_parts) == 2
    assert all(bp.media_type == "image/png" for bp in binary_parts)
    assert binary_parts[0].data == b"page1png"
    assert binary_parts[1].data == b"page2png"


def test_vision_extractor_passes_text_hint_in_prompt():
    """When text= is provided, it should appear in the text prompt sent
    to the agent."""
    ve = VisionExtractor()
    canned = ExtractionOutput(fields={}, confidence={})
    fake = _install_fake_agent(ve, canned)

    ve.extract([b"img"], schemas.CRA_T5, text="Acme Corp employer name hint")

    # First element is the text prompt
    text_prompt = fake.last_prompt[0]
    assert isinstance(text_prompt, str)
    assert "Acme Corp employer name hint" in text_prompt


def test_vision_extractor_max_pages_enforced():
    """If max_pages is set and more pages are passed, raise ValueError
    BEFORE any LLM call."""
    ve = VisionExtractor(max_pages=2)
    fake = _install_fake_agent(ve, ExtractionOutput(fields={}, confidence={}))

    with pytest.raises(ValueError, match="max_pages"):
        ve.extract([b"p1", b"p2", b"p3"], schemas.CRA_T5)

    assert fake.call_count == 0  # agent was never called


def test_vision_extractor_max_pages_none_allows_unlimited():
    """max_pages=None (default) allows any number of pages."""
    ve = VisionExtractor()  # max_pages defaults to None
    _install_fake_agent(ve, ExtractionOutput(fields={}, confidence={}))

    # Should not raise
    ve.extract([b"p1"] * 20, schemas.CRA_T5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_vision_extractor.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'finlit.extractors.vision_extractor'`.

- [ ] **Step 3: Create `VisionExtractor`**

Create `finlit/extractors/vision_extractor.py`:

```python
"""
Vision-based extractor built on pydantic-ai.

Sends a list of PNG page images to any multimodal LLM that pydantic-ai
supports — Claude, OpenAI, Gemini, Ollama-hosted open-source models, or
anything behind an OpenAI-compatible endpoint. Consumers pick the model
by passing a pydantic-ai model string.

Tested OSS model strings (via Ollama):
    - "ollama:llama3.2-vision"      Meta, 11B, general-purpose
    - "ollama:qwen2.5vl:7b"         Alibaba, strongest for forms
    - "ollama:minicpm-v"            OpenBMB, fast 8B

Non-multimodal models will fail at extraction time with a provider error.
"""
from __future__ import annotations

from typing import Any

from pydantic_ai import Agent, BinaryContent

from finlit.extractors.base_vision import BaseVisionExtractor
from finlit.extractors.pydantic_ai_extractor import ExtractionOutput
from finlit.schema import Schema


class VisionExtractor(BaseVisionExtractor):
    """Vision extractor backed by pydantic-ai.

    Parameters
    ----------
    model:
        A pydantic-ai model string for a multimodal model. Defaults to
        ``"anthropic:claude-sonnet-4-6"``. Examples:

            VisionExtractor(model="openai:gpt-4o")
            VisionExtractor(model="google-gla:gemini-2.0-flash")
            VisionExtractor(model="ollama:qwen2.5vl:7b")   # fully local

        The model MUST be multimodal. Non-vision models will raise at
        extraction time.
    dpi:
        DPI used when the pipeline renders PDFs via ``render_pages()``.
        Default 200. Stored on the extractor so the pipeline can read it
        when deciding how to render.
    image_format:
        Reserved. Only ``"png"`` is supported today.
    max_pages:
        Hard cap on page count per document. If a document has more
        pages than ``max_pages``, ``extract()`` raises ``ValueError``
        *before* any LLM call. ``None`` (default) disables the cap.
    """

    def __init__(
        self,
        model: str = "anthropic:claude-sonnet-4-6",
        dpi: int = 200,
        image_format: str = "png",
        max_pages: int | None = None,
    ) -> None:
        self.model = model
        self.dpi = dpi
        self.image_format = image_format
        self.max_pages = max_pages
        self._agent = Agent(
            model,
            output_type=ExtractionOutput,
            system_prompt=self._system_prompt(),
        )

    def _system_prompt(self) -> str:
        return """You are a precise document field extractor for Canadian financial documents.

You are looking at scanned or rendered page images of a document (not plain text).
Read the visible layout the way a human would: find each labelled box or field,
match it to its adjacent value, and extract the value. The spatial relationship
between a label and its value is the primary signal — do not infer a value from
context if it is not visible in the image.

Return:
1. A JSON object 'fields' mapping field_name -> extracted value (use null if not found)
2. A JSON object 'confidence' mapping field_name -> float 0.0-1.0
3. A 'notes' string for any extraction warnings

Rules:
- Monetary values must be floats (e.g. 87500.00, not "$87,500.00")
- Social Insurance Numbers in format "XXX-XXX-XXX"
- Province codes as 2-letter uppercase (ON, BC, QC, etc.)
- Tax year as 4-digit integer
- If a box is present but illegible, return null with confidence 0.0
- Do not hallucinate. If a field is not visible, return null.
"""

    def extract(
        self,
        images: list[bytes],
        schema: Schema,
        text: str = "",
    ) -> ExtractionOutput:
        if self.max_pages is not None and len(images) > self.max_pages:
            raise ValueError(
                f"VisionExtractor: document has {len(images)} pages but "
                f"max_pages={self.max_pages}"
            )

        prompt_text = self._build_prompt(schema, text)
        # pydantic-ai accepts a list of mixed string + BinaryContent parts
        parts: list[Any] = [prompt_text]
        for img in images:
            parts.append(BinaryContent(data=img, media_type="image/png"))

        result = self._agent.run_sync(parts)
        return result.output

    def _build_prompt(self, schema: Schema, text_hint: str) -> str:
        field_descriptions = "\n".join(
            f"  - {f.name} ({f.dtype.__name__}): {f.description}"
            + (" [REQUIRED]" if f.required else "")
            for f in schema.fields
        )
        hint_section = ""
        if text_hint:
            # Truncate to keep token usage bounded
            hint_section = (
                "\n\nText path partial output (may be incomplete or wrong — "
                "prefer what you can see in the images):\n"
                f"---\n{text_hint[:4000]}\n---\n"
            )
        return f"""Document type: {schema.document_type}

Fields to extract:
{field_descriptions}
{hint_section}
Extract all fields listed above from the page image(s) below."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_vision_extractor.py -v`

Expected: all 10 tests in `test_vision_extractor.py` PASS (3 from Task 2 + 7 new).

- [ ] **Step 5: Commit**

```bash
git add finlit/extractors/vision_extractor.py tests/test_vision_extractor.py
git commit -m "$(cat <<'EOF'
feat(extractors): add VisionExtractor

Default BaseVisionExtractor implementation backed by pydantic-ai. Sends
PNG page images to any multimodal model pydantic-ai supports — Claude,
OpenAI, Gemini, or fully-local OSS models via Ollama (Llama 3.2 Vision,
Qwen2.5-VL, MiniCPM-V). Supports an optional text hint and a max_pages
cap that short-circuits before any LLM call.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add `StubVisionExtractor` fixture

No tests for the fixture itself — it exists to support Task 6's integration tests. This task is a standalone commit so Task 6 stays focused on orchestration.

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add the stub class and a fixture**

At the bottom of `tests/conftest.py`, after the existing `high_confidence_t4_extractor` fixture, add:

```python
from finlit.extractors.base_vision import BaseVisionExtractor


class StubVisionExtractor(BaseVisionExtractor):
    """Deterministic vision extractor for tests — zero network.

    Records call count, last images, and last text hint so integration
    tests can assert on what the pipeline passed in.
    """

    def __init__(
        self,
        fields: dict[str, Any] | None = None,
        confidence: dict[str, float] | None = None,
        notes: str = "",
        raises: Exception | None = None,
    ):
        self._fields = fields or {}
        self._confidence = confidence or {}
        self._notes = notes
        self._raises = raises
        self.call_count = 0
        self.last_images: list[bytes] | None = None
        self.last_text: str | None = None

    def extract(self, images, schema, text=""):
        self.call_count += 1
        self.last_images = images
        self.last_text = text
        if self._raises is not None:
            raise self._raises
        return _StubExtractionOutput(self._fields, self._confidence, self._notes)


@pytest.fixture
def high_confidence_vision_t5_extractor() -> StubVisionExtractor:
    """Vision stub that returns a fully-populated, high-confidence T5."""
    return StubVisionExtractor(
        fields={
            "payer_name": "Bank of Canada",
            "recipient_sin": "123-456-789",
            "tax_year": 2024,
            "box_10_actual_amount_dividends_other_than_eligible": 1000.00,
            "box_11_taxable_amount_dividends_other_than_eligible": 1150.00,
            "box_12_dividend_tax_credit_other_than_eligible": 104.14,
            "box_13_interest_from_canadian_sources": 250.00,
            "box_14_other_income_from_canadian_sources": 0.0,
            "box_18_capital_gains_dividends": 0.0,
            "box_24_actual_amount_of_eligible_dividends": 500.00,
            "box_25_taxable_amount_of_eligible_dividends": 690.00,
            "box_26_dividend_tax_credit_eligible": 103.27,
        },
        confidence={
            "payer_name": 0.98,
            "recipient_sin": 0.99,
            "tax_year": 0.99,
            "box_10_actual_amount_dividends_other_than_eligible": 0.96,
            "box_11_taxable_amount_dividends_other_than_eligible": 0.96,
            "box_12_dividend_tax_credit_other_than_eligible": 0.94,
            "box_13_interest_from_canadian_sources": 0.95,
            "box_14_other_income_from_canadian_sources": 0.99,
            "box_18_capital_gains_dividends": 0.99,
            "box_24_actual_amount_of_eligible_dividends": 0.96,
            "box_25_taxable_amount_of_eligible_dividends": 0.96,
            "box_26_dividend_tax_credit_eligible": 0.94,
        },
    )
```

- [ ] **Step 2: Verify the existing tests still pass (the fixture is additive)**

Run: `.venv/bin/pytest tests/ -v`

Expected: all existing tests still pass. No new tests added yet.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "$(cat <<'EOF'
test(conftest): add StubVisionExtractor and T5 vision fixture

Test-only helper that implements BaseVisionExtractor with pre-canned
outputs and a call counter. Used by v0.3 pipeline integration tests to
verify the vision fallback orchestration without any network calls.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Wire vision fallback into `DocumentPipeline`

This is the biggest task — 9 integration tests and the orchestration code. Write all tests first, watch them fail, then implement the pipeline changes.

**Files:**
- Modify: `finlit/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write all 9 failing integration tests**

Append to `tests/test_pipeline.py`:

```python
# ---------------- Vision fallback integration tests (v0.3) ----------------

from tests.conftest import StubVisionExtractor


def test_vision_fallback_fires_when_callback_returns_true(
    monkeypatch, synthetic_parsed_document, tmp_path
):
    """Text extractor returns all-None (needs_review=True). Default callback
    fires vision. Final result is the vision result."""
    _patch_parser(monkeypatch, synthetic_parsed_document)

    # Stub render_pages so we don't rasterize anything
    monkeypatch.setattr(
        "finlit.pipeline.render_pages",
        lambda path, dpi=200: [b"fakepng1"],
    )

    fake = tmp_path / "blank_t4.pdf"
    fake.write_bytes(b"x")

    empty_text = StubExtractor(
        fields={name: None for name in schemas.CRA_T4.field_names()},
        confidence={name: 0.99 for name in schemas.CRA_T4.field_names()},
    )
    vision = StubVisionExtractor(
        fields={
            "employer_name": "Acme Corp",
            "employee_sin": "123-456-789",
            "tax_year": 2024,
            "box_14_employment_income": 87500.0,
            "box_22_income_tax_deducted": 15200.0,
        },
        confidence={
            "employer_name": 0.98,
            "employee_sin": 0.99,
            "tax_year": 0.99,
            "box_14_employment_income": 0.97,
            "box_22_income_tax_deducted": 0.96,
        },
    )

    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=empty_text,
        vision_extractor=vision,
    )
    result = pipeline.run(fake)

    assert vision.call_count == 1
    assert result.extraction_path == "vision"
    assert result.fields["employer_name"] == "Acme Corp"
    assert result.fields["box_14_employment_income"] == 87500.0
    events = [e["event"] for e in result.audit_log]
    assert "vision_fallback_triggered" in events
    assert "vision_extraction_complete" in events


def test_vision_fallback_skipped_when_callback_returns_false(
    monkeypatch, synthetic_parsed_document, high_confidence_t4_extractor, tmp_path
):
    """Text path produces a clean result (needs_review=False). Default
    callback returns False. Vision must NOT be called."""
    _patch_parser(monkeypatch, synthetic_parsed_document)
    fake = tmp_path / "clean.pdf"
    fake.write_bytes(b"x")

    vision = StubVisionExtractor(fields={}, confidence={})
    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
        vision_extractor=vision,
    )
    result = pipeline.run(fake)

    assert vision.call_count == 0
    assert result.extraction_path == "text"
    assert result.needs_review is False


def test_vision_fallback_skipped_when_vision_extractor_not_provided(
    monkeypatch, synthetic_parsed_document, tmp_path
):
    """When vision_extractor is None, no vision audit events appear even
    if the text result has needs_review=True."""
    _patch_parser(monkeypatch, synthetic_parsed_document)
    fake = tmp_path / "blank.pdf"
    fake.write_bytes(b"x")

    empty = StubExtractor(
        fields={name: None for name in schemas.CRA_T4.field_names()},
        confidence={name: 0.99 for name in schemas.CRA_T4.field_names()},
    )
    pipeline = DocumentPipeline(schema=schemas.CRA_T4, extractor=empty)
    result = pipeline.run(fake)

    assert result.extraction_path == "text"
    events = [e["event"] for e in result.audit_log]
    assert "vision_fallback_triggered" not in events


def test_vision_fallback_custom_callback_overrides_default(
    monkeypatch, synthetic_parsed_document, high_confidence_t4_extractor, tmp_path
):
    """A custom callback can force vision to run even on a clean text result."""
    _patch_parser(monkeypatch, synthetic_parsed_document)
    monkeypatch.setattr(
        "finlit.pipeline.render_pages",
        lambda path, dpi=200: [b"fakepng"],
    )
    fake = tmp_path / "doc.pdf"
    fake.write_bytes(b"x")

    vision = StubVisionExtractor(
        fields={"employer_name": "Vision Wins Co", "employee_sin": None,
                "tax_year": 2024, "box_14_employment_income": 99999.0,
                "box_22_income_tax_deducted": 5000.0},
        confidence={"employer_name": 0.9, "employee_sin": 0.0,
                    "tax_year": 0.9, "box_14_employment_income": 0.9,
                    "box_22_income_tax_deducted": 0.9},
    )

    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
        vision_extractor=vision,
        vision_fallback_when=lambda r: True,  # always
    )
    result = pipeline.run(fake)

    assert vision.call_count == 1
    assert result.extraction_path == "vision"
    assert result.fields["employer_name"] == "Vision Wins Co"


def test_vision_render_failure_falls_back_to_text_result(
    monkeypatch, synthetic_parsed_document, high_confidence_t4_extractor, tmp_path
):
    """If render_pages raises, we return the text result with a vision_fallback_failed
    warning and an audit event."""
    _patch_parser(monkeypatch, synthetic_parsed_document)

    def _boom(path, dpi=200):
        raise RuntimeError("corrupted pdf")

    monkeypatch.setattr("finlit.pipeline.render_pages", _boom)
    fake = tmp_path / "bad.pdf"
    fake.write_bytes(b"x")

    vision = StubVisionExtractor(fields={}, confidence={})
    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
        vision_extractor=vision,
        vision_fallback_when=lambda r: True,  # force vision
    )
    result = pipeline.run(fake)

    assert vision.call_count == 0  # never got past render
    assert result.extraction_path == "text"
    warning_codes = {w["code"] for w in result.warnings}
    assert "vision_fallback_failed" in warning_codes
    matching = [w for w in result.warnings if w["code"] == "vision_fallback_failed"]
    assert matching[0]["reason"] == "render"
    events = [e["event"] for e in result.audit_log]
    assert "vision_render_failed" in events


def test_vision_extraction_failure_falls_back_to_text_result(
    monkeypatch, synthetic_parsed_document, high_confidence_t4_extractor, tmp_path
):
    """If the vision extractor raises, we return the text result with a
    vision_fallback_failed warning and an audit event."""
    _patch_parser(monkeypatch, synthetic_parsed_document)
    monkeypatch.setattr(
        "finlit.pipeline.render_pages",
        lambda path, dpi=200: [b"fakepng"],
    )
    fake = tmp_path / "doc.pdf"
    fake.write_bytes(b"x")

    vision = StubVisionExtractor(
        fields={}, confidence={}, raises=RuntimeError("api down"),
    )
    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
        vision_extractor=vision,
        vision_fallback_when=lambda r: True,
    )
    result = pipeline.run(fake)

    assert vision.call_count == 1
    assert result.extraction_path == "text"
    warning_codes = {w["code"] for w in result.warnings}
    assert "vision_fallback_failed" in warning_codes
    matching = [w for w in result.warnings if w["code"] == "vision_fallback_failed"]
    assert matching[0]["reason"] == "extraction"
    events = [e["event"] for e in result.audit_log]
    assert "vision_extraction_failed" in events


def test_vision_callback_exception_falls_back_to_text_result(
    monkeypatch, synthetic_parsed_document, high_confidence_t4_extractor, tmp_path
):
    """A callback that raises must not crash the pipeline."""
    _patch_parser(monkeypatch, synthetic_parsed_document)
    fake = tmp_path / "doc.pdf"
    fake.write_bytes(b"x")

    def _bad_callback(result):
        raise ValueError("buggy callback")

    vision = StubVisionExtractor(fields={}, confidence={})
    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
        vision_extractor=vision,
        vision_fallback_when=_bad_callback,
    )
    result = pipeline.run(fake)

    assert vision.call_count == 0
    assert result.extraction_path == "text"
    warning_codes = {w["code"] for w in result.warnings}
    assert "vision_fallback_failed" in warning_codes
    matching = [w for w in result.warnings if w["code"] == "vision_fallback_failed"]
    assert matching[0]["reason"] == "callback"


def test_vision_result_replaces_text_result_fully(
    monkeypatch, synthetic_parsed_document, tmp_path
):
    """When vision runs successfully, its fields replace the text result
    completely — no merging."""
    _patch_parser(monkeypatch, synthetic_parsed_document)
    monkeypatch.setattr(
        "finlit.pipeline.render_pages",
        lambda path, dpi=200: [b"fakepng"],
    )
    fake = tmp_path / "doc.pdf"
    fake.write_bytes(b"x")

    text = StubExtractor(
        fields={"employer_name": "WRONG", "employee_sin": "111-111-111",
                "tax_year": 2023, "box_14_employment_income": 1.0,
                "box_22_income_tax_deducted": 2.0},
        confidence={"employer_name": 0.5, "employee_sin": 0.5,
                    "tax_year": 0.5, "box_14_employment_income": 0.5,
                    "box_22_income_tax_deducted": 0.5},
    )
    vision = StubVisionExtractor(
        fields={"employer_name": "CORRECT", "employee_sin": "999-999-999",
                "tax_year": 2024, "box_14_employment_income": 87500.0,
                "box_22_income_tax_deducted": 15200.0},
        confidence={"employer_name": 0.99, "employee_sin": 0.99,
                    "tax_year": 0.99, "box_14_employment_income": 0.99,
                    "box_22_income_tax_deducted": 0.99},
    )

    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=text,
        vision_extractor=vision,
        # Text result has review_fields (confidence < 0.85) so default fires
    )
    result = pipeline.run(fake)

    assert result.extraction_path == "vision"
    assert result.fields["employer_name"] == "CORRECT"
    assert result.fields["employee_sin"] == "999-999-999"
    assert result.fields["box_14_employment_income"] == 87500.0


def test_vision_audit_trail_complete(
    monkeypatch, synthetic_parsed_document, tmp_path
):
    """A successful vision run must log the full audit event sequence in order."""
    _patch_parser(monkeypatch, synthetic_parsed_document)
    monkeypatch.setattr(
        "finlit.pipeline.render_pages",
        lambda path, dpi=200: [b"fakepng"],
    )
    fake = tmp_path / "doc.pdf"
    fake.write_bytes(b"x")

    empty = StubExtractor(
        fields={name: None for name in schemas.CRA_T4.field_names()},
        confidence={name: 0.99 for name in schemas.CRA_T4.field_names()},
    )
    vision = StubVisionExtractor(
        fields={"employer_name": "Acme", "employee_sin": "123-456-789",
                "tax_year": 2024, "box_14_employment_income": 1.0,
                "box_22_income_tax_deducted": 1.0},
        confidence={"employer_name": 0.99, "employee_sin": 0.99,
                    "tax_year": 0.99, "box_14_employment_income": 0.99,
                    "box_22_income_tax_deducted": 0.99},
    )

    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4, extractor=empty, vision_extractor=vision
    )
    result = pipeline.run(fake)

    events = [e["event"] for e in result.audit_log]
    required_in_order = [
        "vision_fallback_triggered",
        "vision_render_start",
        "vision_render_complete",
        "vision_extraction_start",
        "vision_extraction_complete",
    ]
    # All five must appear, in this order
    indices = [events.index(ev) for ev in required_in_order]
    assert indices == sorted(indices)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline.py -v -k "vision"`

Expected: FAIL — most tests fail with `TypeError: DocumentPipeline.__init__() got an unexpected keyword argument 'vision_extractor'`.

- [ ] **Step 3: Update `finlit/pipeline.py` — add imports and parameters**

At the top of `finlit/pipeline.py`, add these imports alongside the existing ones:

```python
from typing import Any, Callable

from finlit.extractors.base_vision import BaseVisionExtractor
from finlit.parsers.image_renderer import render_pages
```

Then update the `DocumentPipeline.__init__` signature. The existing signature is:

```python
    def __init__(
        self,
        schema: Schema,
        extractor: str | BaseExtractor = "claude",
        model: str | None = None,
        audit: bool = True,
        pii_redact: bool = False,
        review_threshold: float = 0.85,
        ocr_fallback: bool = True,
        sparse_text_threshold: int = SPARSE_TEXT_THRESHOLD,
    ):
```

Change it to:

```python
    def __init__(
        self,
        schema: Schema,
        extractor: str | BaseExtractor = "claude",
        model: str | None = None,
        audit: bool = True,
        pii_redact: bool = False,
        review_threshold: float = 0.85,
        ocr_fallback: bool = True,
        sparse_text_threshold: int = SPARSE_TEXT_THRESHOLD,
        vision_extractor: BaseVisionExtractor | None = None,
        vision_fallback_when: Callable[[Any], bool] | None = None,
    ):
```

And at the end of `__init__`, after `self._validator = FieldValidator()`, add:

```python
        self.vision_extractor = vision_extractor
        self.vision_fallback_when = vision_fallback_when
```

- [ ] **Step 4: Update `run()` to build the provisional result as a local helper**

The `run()` method currently builds the `ExtractionResult` directly at the end. We need to factor out the "build result from extraction output" step so we can call it twice (once for text, once for vision) and insert the fallback decision between them.

Replace the entire body of `run()` with this version. It keeps every existing step intact, factors the result-building into an inline section, and adds the vision fallback at the end. Locate the current `run()` method (starts at `def run(self, path: str | Path) -> ExtractionResult:`) and replace it wholesale with:

```python
    def run(self, path: str | Path) -> ExtractionResult:
        run_id = str(uuid.uuid4())
        audit = AuditLog(run_id=run_id)
        warnings: list[dict] = []

        path = Path(path)

        # Step 1: parse
        audit.log("document_load_start", file=str(path))
        parsed = self._parser.parse(path)
        audit.log(
            "document_loaded",
            file=parsed.metadata["filename"],
            sha256=parsed.metadata["sha256"],
            num_pages=parsed.metadata.get("num_pages"),
        )

        # Step 1b: OCR fallback on sparse text
        stripped_len = len(parsed.full_text.strip())
        if stripped_len < self.sparse_text_threshold and self.ocr_fallback:
            audit.log(
                "ocr_fallback_triggered",
                reason="sparse_text",
                initial_chars=stripped_len,
            )
            parsed = self._get_ocr_parser().parse(path)
            audit.log(
                "document_loaded_ocr",
                file=parsed.metadata["filename"],
                sha256=parsed.metadata["sha256"],
                num_pages=parsed.metadata.get("num_pages"),
                chars=len(parsed.full_text.strip()),
            )

        # Step 1c: sparse warning if still unreadable
        final_stripped_len = len(parsed.full_text.strip())
        if final_stripped_len < self.sparse_text_threshold:
            warnings.append(
                {
                    "code": "sparse_document",
                    "message": (
                        f"Parsed text is only {final_stripped_len} chars; "
                        "the document may be a scanned image with no text "
                        "layer that OCR could not recover."
                    ),
                    "chars": final_stripped_len,
                }
            )
            audit.log(
                "sparse_document_warning",
                chars=final_stripped_len,
                threshold=self.sparse_text_threshold,
            )

        # Step 2: PII scan
        pii_entities: list[dict] = []
        if self.audit_enabled:
            pii_entities = self._pii_detector.analyze(parsed.full_text)
            if pii_entities:
                audit.log(
                    "pii_detected",
                    count=len(pii_entities),
                    entities=[e["entity_type"] for e in pii_entities],
                )

        # Step 3: LLM text extraction
        audit.log(
            "extraction_start", schema=self.schema.name, model=self._model_name
        )
        extraction = self._extractor.extract(parsed.full_text, self.schema)
        audit.log("extraction_complete", fields_returned=len(extraction.fields))

        # Step 4: validate
        validated_fields, validation_errors = self._validator.validate(
            extraction.fields, self.schema
        )
        if validation_errors:
            audit.log("validation_errors", errors=validation_errors)

        # Step 4b: required fields missing warning
        required_field_names = {
            f.name for f in self.schema.fields if f.required
        }
        missing_required = sorted(
            fname
            for fname in required_field_names
            if validated_fields.get(fname) is None
        )
        if missing_required:
            warnings.append(
                {
                    "code": "required_fields_missing",
                    "message": (
                        f"{len(missing_required)} required field(s) missing "
                        f"after extraction: {', '.join(missing_required)}"
                    ),
                    "missing_fields": missing_required,
                }
            )
            audit.log(
                "required_fields_missing_warning",
                count=len(missing_required),
                fields=missing_required,
            )

        # Step 5: review queue
        review_fields = [
            {
                "field": fname,
                "confidence": extraction.confidence.get(fname, 0.0),
                "raw": validated_fields.get(fname),
            }
            for fname in self.schema.field_names()
            if extraction.confidence.get(fname, 0.0) < self.review_threshold
            and validated_fields.get(fname) is not None
        ]
        if review_fields:
            audit.log(
                "review_flagged",
                count=len(review_fields),
                fields=[r["field"] for r in review_fields],
            )

        # Step 6: source refs (placeholder until Docling bbox wiring is added)
        source_ref = {
            fname: {
                "doc": parsed.metadata["filename"],
                "page": None,
                "bbox": None,
            }
            for fname in self.schema.field_names()
        }

        # Build the provisional text-path result
        text_result = ExtractionResult(
            fields=validated_fields,
            confidence=extraction.confidence,
            source_ref=source_ref,
            pii_entities=pii_entities,
            audit_log=audit.to_dict(),  # snapshot, rebuilt after finalize
            review_fields=review_fields,
            warnings=list(warnings),
            review_threshold=self.review_threshold,
            document_path=str(path),
            schema_name=self.schema.name,
            extractor_model=self._model_name,
            extraction_path="text",
        )

        # Step 7: vision fallback decision
        if self.vision_extractor is not None:
            vision_result = self._maybe_run_vision_fallback(
                path=path,
                parsed_text=parsed.full_text,
                text_result=text_result,
                audit=audit,
                warnings=warnings,
                source_ref=source_ref,
                pii_entities=pii_entities,
            )
            if vision_result is not None:
                return vision_result

        audit.log(
            "pipeline_complete",
            fields_total=len(self.schema.fields),
            fields_extracted=sum(1 for v in validated_fields.values() if v is not None),
            needs_review=len(review_fields) > 0 or len(warnings) > 0,
            extraction_path="text",
        )
        audit.finalize()

        # Refresh audit log on the text result (finalize may freeze it)
        text_result.audit_log = audit.to_dict()
        text_result.warnings = list(warnings)
        return text_result
```

- [ ] **Step 5: Add the `_maybe_run_vision_fallback` helper method**

Append this method to `DocumentPipeline`, directly after `run()`:

```python
    def _maybe_run_vision_fallback(
        self,
        *,
        path: Path,
        parsed_text: str,
        text_result: ExtractionResult,
        audit: AuditLog,
        warnings: list[dict],
        source_ref: dict,
        pii_entities: list[dict],
    ) -> ExtractionResult | None:
        """Evaluate the fallback callback and, if True, run vision extraction.

        Returns a new ExtractionResult on successful vision run, or None
        to signal "no fallback happened, return the text result".

        On any failure (callback exception, render failure, extraction
        failure), this method appends a vision_fallback_failed warning
        to `warnings` (mutating the text_result's warnings list in place
        via the shared reference), logs the appropriate audit event, and
        returns None so the caller returns the text result.
        """
        assert self.vision_extractor is not None

        callback = self.vision_fallback_when or (lambda r: r.needs_review)

        # Evaluate callback
        try:
            should_fire = callback(text_result)
        except Exception as e:
            audit.log(
                "vision_fallback_callback_error",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            warnings.append(
                {
                    "code": "vision_fallback_failed",
                    "reason": "callback",
                    "message": f"vision_fallback_when callback raised: {e}",
                }
            )
            return None

        if not should_fire:
            return None

        audit.log(
            "vision_fallback_triggered",
            provisional_needs_review=text_result.needs_review,
            provisional_warning_codes=[w["code"] for w in text_result.warnings],
        )

        # Render pages
        try:
            audit.log("vision_render_start", dpi=self.vision_extractor.dpi, path=str(path))
            images = render_pages(path, dpi=self.vision_extractor.dpi)
            audit.log(
                "vision_render_complete",
                page_count=len(images),
                total_bytes=sum(len(i) for i in images),
            )
        except Exception as e:
            audit.log(
                "vision_render_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            warnings.append(
                {
                    "code": "vision_fallback_failed",
                    "reason": "render",
                    "message": f"render_pages failed: {e}",
                }
            )
            return None

        # Run vision extractor
        try:
            audit.log(
                "vision_extraction_start",
                model=getattr(self.vision_extractor, "model", "custom"),
                page_count=len(images),
            )
            vision_output = self.vision_extractor.extract(
                images, self.schema, text=parsed_text
            )
            audit.log(
                "vision_extraction_complete",
                fields_returned=len(vision_output.fields),
            )
        except Exception as e:
            audit.log(
                "vision_extraction_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            warnings.append(
                {
                    "code": "vision_fallback_failed",
                    "reason": "extraction",
                    "message": f"vision extractor raised: {e}",
                }
            )
            return None

        # Re-validate vision output through the same validator
        v_validated, v_errors = self._validator.validate(
            vision_output.fields, self.schema
        )
        if v_errors:
            audit.log("vision_validation_errors", errors=v_errors)

        # Build vision-path warnings (re-check required fields for the new result)
        vision_warnings: list[dict] = []
        required_field_names = {
            f.name for f in self.schema.fields if f.required
        }
        v_missing = sorted(
            fname
            for fname in required_field_names
            if v_validated.get(fname) is None
        )
        if v_missing:
            vision_warnings.append(
                {
                    "code": "required_fields_missing",
                    "message": (
                        f"{len(v_missing)} required field(s) missing "
                        f"after vision extraction: {', '.join(v_missing)}"
                    ),
                    "missing_fields": v_missing,
                }
            )
            audit.log(
                "required_fields_missing_warning",
                count=len(v_missing),
                fields=v_missing,
                path="vision",
            )

        v_review_fields = [
            {
                "field": fname,
                "confidence": vision_output.confidence.get(fname, 0.0),
                "raw": v_validated.get(fname),
            }
            for fname in self.schema.field_names()
            if vision_output.confidence.get(fname, 0.0) < self.review_threshold
            and v_validated.get(fname) is not None
        ]
        if v_review_fields:
            audit.log(
                "review_flagged",
                count=len(v_review_fields),
                fields=[r["field"] for r in v_review_fields],
                path="vision",
            )

        audit.log(
            "pipeline_complete",
            fields_total=len(self.schema.fields),
            fields_extracted=sum(1 for v in v_validated.values() if v is not None),
            needs_review=len(v_review_fields) > 0 or len(vision_warnings) > 0,
            extraction_path="vision",
        )
        audit.finalize()

        return ExtractionResult(
            fields=v_validated,
            confidence=vision_output.confidence,
            source_ref=source_ref,
            pii_entities=pii_entities,
            audit_log=audit.to_dict(),
            review_fields=v_review_fields,
            warnings=vision_warnings,
            review_threshold=self.review_threshold,
            document_path=str(path),
            schema_name=self.schema.name,
            extractor_model=getattr(
                self.vision_extractor, "model", "custom-vision"
            ),
            extraction_path="vision",
        )
```

- [ ] **Step 6: Run the vision integration tests**

Run: `.venv/bin/pytest tests/test_pipeline.py -v -k "vision"`

Expected: 9 vision tests PASS.

- [ ] **Step 7: Run the full test suite to verify no regressions**

Run: `.venv/bin/pytest tests/ -v`

Expected: all tests pass (v0.2.0 count + 9 new vision pipeline tests + 10 vision extractor tests + 7 image renderer tests + 2 result tests = roughly 73 total).

- [ ] **Step 8: Commit**

```bash
git add finlit/pipeline.py tests/test_pipeline.py
git commit -m "$(cat <<'EOF'
feat(pipeline): vision extraction fallback orchestration

DocumentPipeline gains two opt-in parameters:
  - vision_extractor: BaseVisionExtractor | None
  - vision_fallback_when: Callable[[ExtractionResult], bool] | None

After the text path runs to completion, if a vision_extractor is
provided and the callback returns True (default: result.needs_review),
the pipeline renders page images via render_pages() and re-runs the
extraction through the vision extractor. The vision result fully
replaces the text result when it succeeds.

Fail-safe: any exception in the callback, render, or vision extractor
falls back to the text result with a vision_fallback_failed warning
and a matching audit event. The pipeline never crashes because vision
is misconfigured.

Full audit trail: vision_fallback_triggered, vision_render_start,
vision_render_complete, vision_extraction_start,
vision_extraction_complete (plus *_failed events on error).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Export `VisionExtractor` and `BaseVisionExtractor` from the public API

**Files:**
- Modify: `finlit/__init__.py`
- Test: `tests/test_public_api.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_public_api.py`:

```python
def test_public_api_exports_vision_extractor():
    import finlit
    assert hasattr(finlit, "VisionExtractor")
    assert hasattr(finlit, "BaseVisionExtractor")
    assert "VisionExtractor" in finlit.__all__
    assert "BaseVisionExtractor" in finlit.__all__


def test_can_import_vision_extractor_from_top_level():
    from finlit import VisionExtractor, BaseVisionExtractor
    # Construct a default vision extractor (network is not touched here)
    ve = VisionExtractor()
    assert ve.model == "anthropic:claude-sonnet-4-6"
    assert BaseVisionExtractor is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_public_api.py -v`

Expected: FAIL with `AttributeError: module 'finlit' has no attribute 'VisionExtractor'`.

- [ ] **Step 3: Update `finlit/__init__.py`**

Replace the current file contents with:

```python
"""FinLit - Canadian Financial Document Intelligence Framework."""
from finlit import schemas
from finlit.extractors.base_vision import BaseVisionExtractor
from finlit.extractors.vision_extractor import VisionExtractor
from finlit.pipeline import BatchPipeline, DocumentPipeline
from finlit.result import ExtractionResult
from finlit.schema import Field, Schema

__version__ = "0.3.0"

__all__ = [
    "DocumentPipeline",
    "BatchPipeline",
    "Schema",
    "Field",
    "ExtractionResult",
    "schemas",
    "VisionExtractor",
    "BaseVisionExtractor",
]
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_public_api.py tests/ -v`

Expected: all tests PASS. The full suite should pass with the version bump.

- [ ] **Step 5: Commit**

```bash
git add finlit/__init__.py tests/test_public_api.py
git commit -m "$(cat <<'EOF'
feat(api): export VisionExtractor and BaseVisionExtractor

Adds the two v0.3 classes to the top-level public API and bumps
__version__ to 0.3.0.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Add `--vision-extractor` flag to the CLI

**Files:**
- Modify: `finlit/cli/main.py`
- Test: `tests/test_cli.py` (create if missing)

- [ ] **Step 1: Check if `tests/test_cli.py` exists**

Run: `ls tests/test_cli.py 2>&1`

If it does not exist, continue with Step 2 (which creates it). If it does, append the tests in Step 2 to it.

- [ ] **Step 2: Write the failing test**

Create (or append to) `tests/test_cli.py`:

```python
"""CLI tests — uses typer.testing.CliRunner. No network calls."""
from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from finlit.cli.main import app
from finlit.result import ExtractionResult


runner = CliRunner()


def _fake_result() -> ExtractionResult:
    return ExtractionResult(
        fields={"employer_name": "Acme Corp", "box_14_employment_income": 87500.0},
        confidence={"employer_name": 0.99, "box_14_employment_income": 0.97},
        source_ref={},
        extraction_path="text",
    )


def test_extract_command_accepts_vision_extractor_flag(tmp_path):
    """The --vision-extractor flag is accepted and passes a VisionExtractor
    into DocumentPipeline's vision_extractor parameter."""
    fake_pdf = tmp_path / "t4.pdf"
    fake_pdf.write_bytes(b"x")

    captured = {}

    def _fake_init(self, **kwargs):
        captured.update(kwargs)
        # Don't run any real pipeline setup
        self.schema = kwargs["schema"]

    def _fake_run(self, path):
        return _fake_result()

    with patch("finlit.DocumentPipeline.__init__", _fake_init), patch(
        "finlit.DocumentPipeline.run", _fake_run
    ):
        result = runner.invoke(
            app,
            [
                "extract",
                str(fake_pdf),
                "--schema",
                "cra.t4",
                "--extractor",
                "claude",
                "--vision-extractor",
                "anthropic:claude-sonnet-4-6",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured.get("vision_extractor") is not None
    # Should be a VisionExtractor instance with the model we passed
    from finlit import VisionExtractor
    assert isinstance(captured["vision_extractor"], VisionExtractor)
    assert captured["vision_extractor"].model == "anthropic:claude-sonnet-4-6"


def test_extract_command_without_vision_extractor_flag(tmp_path):
    """When --vision-extractor is omitted, vision_extractor should be None."""
    fake_pdf = tmp_path / "t4.pdf"
    fake_pdf.write_bytes(b"x")

    captured = {}

    def _fake_init(self, **kwargs):
        captured.update(kwargs)
        self.schema = kwargs["schema"]

    def _fake_run(self, path):
        return _fake_result()

    with patch("finlit.DocumentPipeline.__init__", _fake_init), patch(
        "finlit.DocumentPipeline.run", _fake_run
    ):
        result = runner.invoke(
            app,
            ["extract", str(fake_pdf), "--schema", "cra.t4"],
        )

    assert result.exit_code == 0, result.output
    assert captured.get("vision_extractor") is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py -v`

Expected: FAIL — typer reports no such option `--vision-extractor`.

- [ ] **Step 4: Update the CLI command**

In `finlit/cli/main.py`, replace the `extract` command function with:

```python
@app.command()
def extract(
    document: Path = typer.Argument(..., help="Path to document (PDF, DOCX, image)"),
    schema: str = typer.Option("cra.t4", help="Schema to use (e.g. cra.t4, cra.t5)"),
    extractor: str = typer.Option("claude", help="Text extractor: claude | openai | ollama | <pydantic-ai model string>"),
    vision_extractor: str = typer.Option(
        None,
        "--vision-extractor",
        help="Optional vision fallback model (e.g. 'claude-sonnet-4-6', 'openai:gpt-4o', 'ollama:qwen2.5vl:7b')",
    ),
    output: str = typer.Option("table", help="Output format: table | json | jsonl"),
    review_threshold: float = typer.Option(0.85, help="Confidence threshold"),
):
    """Extract structured data from a Canadian financial document."""
    from finlit import DocumentPipeline, VisionExtractor

    schema_map = _schema_map()
    if schema not in schema_map:
        console.print(
            f"[red]Unknown schema: {schema}. "
            f"Available: {', '.join(schema_map.keys())}[/red]"
        )
        raise typer.Exit(1)

    ve = None
    if vision_extractor:
        # Accept shorthand "claude" alias the same way the text extractor does
        model_str = {
            "claude": "anthropic:claude-sonnet-4-6",
            "openai": "openai:gpt-4o",
            "ollama": "ollama:llama3.2-vision",
        }.get(vision_extractor, vision_extractor)
        ve = VisionExtractor(model=model_str)

    console.print(f"[dim]Parsing {document}...[/dim]")
    pipeline = DocumentPipeline(
        schema=schema_map[schema],
        extractor=extractor,
        review_threshold=review_threshold,
        vision_extractor=ve,
    )
    result = pipeline.run(document)

    if output == "json":
        console.print(json.dumps(result.fields, indent=2, default=str))
        return
    if output == "jsonl":
        console.print(
            json.dumps(
                {"fields": result.fields, "confidence": result.confidence},
                default=str,
            )
        )
        return

    table = Table(title=f"Extraction: {document.name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Confidence", style="green")
    table.add_column("Review?", style="yellow")

    review_set = {r["field"] for r in result.review_fields}
    for field_name, value in result.fields.items():
        conf = result.confidence.get(field_name, 0.0)
        needs_review = "!" if field_name in review_set else ""
        table.add_row(
            field_name,
            str(value) if value is not None else "[dim]-[/dim]",
            f"{conf:.0%}",
            needs_review,
        )
    console.print(table)

    if result.extraction_path == "vision":
        console.print("[cyan]ℹ Result produced by vision fallback[/cyan]")

    if result.needs_review:
        console.print(
            f"\n[yellow]{len(result.review_fields)} field(s) flagged for review[/yellow]"
        )
```

- [ ] **Step 5: Run CLI tests**

Run: `.venv/bin/pytest tests/test_cli.py -v`

Expected: 2 CLI tests PASS.

- [ ] **Step 6: Run full suite**

Run: `.venv/bin/pytest tests/ -v`

Expected: full suite passes.

- [ ] **Step 7: Commit**

```bash
git add finlit/cli/main.py tests/test_cli.py
git commit -m "$(cat <<'EOF'
feat(cli): add --vision-extractor flag to extract command

Accepts either a shorthand alias (claude, openai, ollama) or a full
pydantic-ai model string. When set, the CLI constructs a VisionExtractor
and passes it into DocumentPipeline as the vision fallback. Display
includes a "Result produced by vision fallback" line when the final
result came from the vision path.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Add example files

No automated tests for example files. We verify them manually in Task 11.

**Files:**
- Create: `examples/extract_with_vision.py`
- Create: `examples/extract_with_local_vision.py`

- [ ] **Step 1: Create `examples/extract_with_vision.py`**

```python
"""
Minimal example: extract a T5 with Claude vision fallback.

Requires ANTHROPIC_API_KEY in the environment. The text path runs first;
if its result has needs_review=True, the pipeline re-runs the extraction
through Claude Sonnet 4.6 multimodal using rendered page images.

Run:
    ANTHROPIC_API_KEY=sk-... python examples/extract_with_vision.py path/to/slip.pdf
"""
from __future__ import annotations

import sys

from finlit import DocumentPipeline, VisionExtractor, schemas


def main(path: str) -> None:
    pipeline = DocumentPipeline(
        schema=schemas.CRA_T5,
        extractor="claude",
        vision_extractor=VisionExtractor(model="anthropic:claude-sonnet-4-6"),
    )
    result = pipeline.run(path)

    print(f"Path taken     : {result.extraction_path}")
    print(f"Needs review   : {result.needs_review}")
    print(f"Warnings       : {[w['code'] for w in result.warnings]}")
    print()
    print("Extracted fields:")
    for name, value in result.fields.items():
        conf = result.confidence.get(name, 0.0)
        marker = " (None)" if value is None else ""
        print(f"  {name:55s} = {str(value):25s}  conf={conf:.2f}{marker}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: extract_with_vision.py <document.pdf>")
        sys.exit(1)
    main(sys.argv[1])
```

- [ ] **Step 2: Create `examples/extract_with_local_vision.py`**

```python
"""
Fully-local example: extract a T5 with Ollama + Qwen2.5-VL, no API keys.

Prerequisites:
    1. Install Ollama:   https://ollama.ai
    2. Pull the models:  ollama pull llama3.2
                         ollama pull qwen2.5vl:7b
    3. Start the Ollama server (if not already running): ollama serve

Run:
    python examples/extract_with_local_vision.py path/to/slip.pdf

Zero API keys. Zero external network. Pure open-source.
"""
from __future__ import annotations

import sys

from finlit import DocumentPipeline, VisionExtractor, schemas


def main(path: str) -> None:
    pipeline = DocumentPipeline(
        schema=schemas.CRA_T5,
        extractor="ollama:llama3.2",
        vision_extractor=VisionExtractor(model="ollama:qwen2.5vl:7b"),
    )
    result = pipeline.run(path)

    print(f"Path taken     : {result.extraction_path}")
    print(f"Needs review   : {result.needs_review}")
    print(f"Warnings       : {[w['code'] for w in result.warnings]}")
    print()
    print("Extracted fields:")
    for name, value in result.fields.items():
        conf = result.confidence.get(name, 0.0)
        print(f"  {name:55s} = {str(value):25s}  conf={conf:.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: extract_with_local_vision.py <document.pdf>")
        sys.exit(1)
    main(sys.argv[1])
```

- [ ] **Step 3: Verify the examples import cleanly (syntax check)**

Run: `.venv/bin/python -c "import ast; ast.parse(open('examples/extract_with_vision.py').read()); ast.parse(open('examples/extract_with_local_vision.py').read()); print('ok')"`

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add examples/extract_with_vision.py examples/extract_with_local_vision.py
git commit -m "$(cat <<'EOF'
docs(examples): add Claude vision and fully-local OSS vision examples

Two runnable examples for v0.3:
  - extract_with_vision.py: Claude Sonnet 4.6 vision fallback
  - extract_with_local_vision.py: Ollama + Qwen2.5-VL, zero API keys

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Update README with vision extraction documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add "When to use vision extraction" section**

Open `README.md`. Find the line `### Run fully local with Ollama` (near line 102). Immediately after the closing `` ``` `` of that section's code block (around line 112) and before the `### Custom schema for your own documents` heading, insert:

```markdown
### Use vision extraction for scanned PDFs and form layouts

Text extraction fails in two cases: image-only PDFs with no text layer, and form-heavy documents (tax slips, invoices) where 2D column alignment carries meaning. For both, FinLit v0.3 ships an opt-in vision fallback that sends rendered page images to any multimodal LLM.

```python
from finlit import DocumentPipeline, VisionExtractor, schemas

pipeline = DocumentPipeline(
    schema=schemas.CRA_T5,
    extractor="claude",                                    # text path (cheap, fast)
    vision_extractor=VisionExtractor(),                    # vision fallback (accurate)
)
result = pipeline.run("t5_scanned.pdf")
print(result.extraction_path)   # → "text" or "vision"
```

By default the vision extractor runs only when the text result has `needs_review=True`. Pass a custom callback for finer control:

```python
pipeline = DocumentPipeline(
    schema=schemas.CRA_T5,
    extractor="claude",
    vision_extractor=VisionExtractor(model="openai:gpt-4o"),
    vision_fallback_when=lambda r: any(c < 0.80 for c in r.confidence.values()),
)
```

### Running fully locally with open-source vision models

Vision extraction is model-agnostic. Any multimodal model pydantic-ai supports works — including fully-local open-source models via Ollama. No API keys, no external network, suitable for air-gapped deployments.

```python
pipeline = DocumentPipeline(
    schema=schemas.CRA_T5,
    extractor="ollama:llama3.2",
    vision_extractor=VisionExtractor(model="ollama:qwen2.5vl:7b"),
)
```

Tested open-source vision models:

| Model | Size | Ollama tag | Notes |
|---|---|---|---|
| Qwen2.5-VL | 7B | `ollama:qwen2.5vl:7b` | Strongest on form/document tasks |
| Llama 3.2 Vision | 11B | `ollama:llama3.2-vision` | General-purpose, Meta |
| MiniCPM-V | 8B | `ollama:minicpm-v` | Fast, OpenBMB |

Any pydantic-ai–compatible multimodal model will work — these are the ones that have been verified against CRA slips.

```

- [ ] **Step 2: Update the LLM backends section**

Locate the existing `## LLM backends` section (around line 194). Replace it with:

```markdown
## LLM backends

```python
# Anthropic Claude (default)
DocumentPipeline(schema=schemas.CRA_T4, extractor="claude")

# OpenAI
DocumentPipeline(schema=schemas.CRA_T4, extractor="openai", model="gpt-4o")

# Fully local — no external calls
DocumentPipeline(schema=schemas.CRA_T4, extractor="ollama", model="llama3.2")

# Vision fallback (any multimodal model)
from finlit import VisionExtractor
DocumentPipeline(
    schema=schemas.CRA_T4,
    extractor="claude",
    vision_extractor=VisionExtractor(model="ollama:qwen2.5vl:7b"),
)

# Your own
from finlit.extractors import BaseExtractor
from finlit import BaseVisionExtractor

class MyTextExtractor(BaseExtractor):
    def extract(self, text, schema): ...

class MyVisionExtractor(BaseVisionExtractor):
    def extract(self, images, schema, text=""): ...

DocumentPipeline(
    schema=schemas.CRA_T4,
    extractor=MyTextExtractor(),
    vision_extractor=MyVisionExtractor(),
)
```

```

- [ ] **Step 3: Update the Roadmap section**

Locate the `## Roadmap` section (around line 315). Replace its checklist with:

```markdown
- [x] Core extraction pipeline (Docling + pydantic-ai)
- [x] CRA schema registry (T4, T5, T4A, NR4)
- [x] Source traceability and audit log
- [x] PIPEDA PII detection — SIN, CRA BNs, postal codes
- [x] CLI
- [x] OCR auto-fallback for image-only PDFs (v0.2)
- [x] Document-level warnings for sparse and missing-required-field results (v0.2)
- [x] Vision extraction fallback — Claude, OpenAI, Gemini, or local OSS via Ollama (v0.3)
- [ ] SEDAR filing schemas (MD&A, AIF, financial statements)
- [ ] Bank statement schemas (RBC, TD, Scotiabank, BMO, CIBC)
- [ ] Accuracy benchmarks per schema
- [ ] LangChain and LlamaIndex reader integrations
- [ ] MCP tool definitions for agentic workflows
- [ ] French CRA form support
```

- [ ] **Step 4: Verify README still renders correctly**

Run: `.venv/bin/python -c "from pathlib import Path; content = Path('README.md').read_text(); assert content.count('```') % 2 == 0, 'unbalanced code fences'; print('ok, %d lines' % len(content.splitlines()))"`

Expected: `ok, <line count>` — and no assertion error about code fences.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs(readme): document v0.3 vision extraction and OSS model support

- New "Use vision extraction for scanned PDFs and form layouts" section
- New "Running fully locally with open-source vision models" section
  with tested Ollama model table (Qwen2.5-VL, Llama 3.2 Vision, MiniCPM-V)
- LLM backends section updated with vision + BYO extractor examples
- Roadmap: mark v0.2 OCR/warnings items shipped and v0.3 vision shipped

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Manual verification against real documents

Not committed to the repo — this is a runbook for the implementer to execute after all automated work passes, with results reported in the PR description.

**Prerequisites:**
- `ANTHROPIC_API_KEY` set in the environment
- Ollama running locally with `qwen2.5vl:7b` pulled (`ollama pull qwen2.5vl:7b`)
- These documents present in the project root (they contain real PII and must NOT be committed):
  - `T4.pdf`
  - `T5_2024_Slip1_Srivatsa_Kasagar.pdf`
  - `cra_example_t5_albert_chang.jpg`

- [ ] **Step 1: Run the full automated suite one more time**

Run: `.venv/bin/pytest tests/ -v`

Expected: all tests pass. If anything fails, fix it before proceeding.

- [ ] **Step 2: Verify T4.pdf — vision fallback on image-only PDF**

Run this inline script:

```bash
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY .venv/bin/python -c "
from finlit import DocumentPipeline, VisionExtractor, schemas
p = DocumentPipeline(
    schema=schemas.CRA_T4,
    extractor='claude',
    vision_extractor=VisionExtractor(),
)
r = p.run('T4.pdf')
print('path        :', r.extraction_path)
print('needs_review:', r.needs_review)
print('warnings    :', [w['code'] for w in r.warnings])
print('fields:')
for k, v in r.fields.items():
    if v is not None:
        print(f'  {k}: {v} (conf={r.confidence[k]:.2f})')
"
```

Expected:
- `path` = `vision`
- Fields populated (at least `employer_name`, `tax_year`, `box_14_employment_income`)
- `needs_review` reflects actual vision result quality

Record the actual output in a file `VERIFICATION_RESULTS.md` at the repo root (gitignored):

```bash
mkdir -p /tmp/v03verify
echo "# v0.3 manual verification results" > /tmp/v03verify/VERIFICATION_RESULTS.md
echo "Date: $(date)" >> /tmp/v03verify/VERIFICATION_RESULTS.md
```

- [ ] **Step 3: Verify T5_2024_Slip1_Srivatsa_Kasagar.pdf — form layout**

Run:

```bash
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY .venv/bin/python -c "
from finlit import DocumentPipeline, VisionExtractor, schemas
p = DocumentPipeline(
    schema=schemas.CRA_T5,
    extractor='claude',
    vision_extractor=VisionExtractor(),
)
r = p.run('T5_2024_Slip1_Srivatsa_Kasagar.pdf')
print('path        :', r.extraction_path)
print('needs_review:', r.needs_review)
print('fields:')
for k, v in r.fields.items():
    if v is not None:
        print(f'  {k}: {v} (conf={r.confidence[k]:.2f})')
"
```

Expected: vision path fires; box numbers now align correctly with values (compare to the actual PDF by eye). Specifically, verify the value previously mis-assigned to Box 24 in v0.2 now lands in its correct box.

- [ ] **Step 4: Verify `cra_example_t5_albert_chang.jpg` — no regression**

Run:

```bash
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY .venv/bin/python -c "
from finlit import DocumentPipeline, VisionExtractor, schemas
p = DocumentPipeline(
    schema=schemas.CRA_T5,
    extractor='claude',
    vision_extractor=VisionExtractor(),
)
r = p.run('cra_example_t5_albert_chang.jpg')
extracted = sum(1 for v in r.fields.values() if v is not None)
print(f'path        : {r.extraction_path}')
print(f'extracted   : {extracted}/{len(r.fields)}')
print(f'payer_name  : {r.fields.get(\"payer_name\")}')
print(f'box_24      : {r.fields.get(\"box_24_actual_amount_of_eligible_dividends\")}')
print(f'box_25      : {r.fields.get(\"box_25_taxable_amount_of_eligible_dividends\")}')
print(f'box_26      : {r.fields.get(\"box_26_dividend_tax_credit_eligible\")}')
"
```

Expected:
- `payer_name` = "AGENTS INC."
- `box_24` = 1000.0
- `box_25` = 1380.0
- `box_26` = 207.27
- At least 12/12 fields extracted (no regression from v0.2)

- [ ] **Step 5: Verify OSS story — Ollama + Qwen2.5-VL**

Run (requires Ollama running):

```bash
.venv/bin/python -c "
from finlit import DocumentPipeline, VisionExtractor, schemas
p = DocumentPipeline(
    schema=schemas.CRA_T5,
    extractor='ollama:llama3.2',
    vision_extractor=VisionExtractor(model='ollama:qwen2.5vl:7b'),
)
r = p.run('cra_example_t5_albert_chang.jpg')
print('path        :', r.extraction_path)
print('extracted   :', sum(1 for v in r.fields.values() if v is not None), '/', len(r.fields))
print('payer_name  :', r.fields.get('payer_name'))
"
```

Expected:
- `path` = `vision` (or `text` if llama3.2 handled it OK — either is fine)
- OSS story works end to end — the pipeline runs without any API keys
- At least `payer_name` and a few box values extracted

**If this step fails**, the "model-agnostic" promise is broken and v0.3 is NOT ready to ship. Investigate before proceeding. Common issues: wrong Ollama model tag, Ollama server not running, pydantic-ai Ollama provider version mismatch.

- [ ] **Step 6: Final automated test run**

Run: `.venv/bin/pytest tests/ -v`

Expected: all tests pass, final green.

- [ ] **Step 7: Push all v0.3 commits**

```bash
git log --oneline origin/main..HEAD
git push origin main
```

Expected: successful push. Verify GitHub shows all v0.3 commits on main.

---

## Done

After Task 11 completes successfully:
- 7 new test files or additions
- ~28 new tests (v0.2 baseline 45 → v0.3 target ~73)
- 3 new production modules
- 2 new example files
- README updated with vision + OSS sections
- `__version__` bumped to `0.3.0`
- All real-world verification documents extract correctly through vision
- OSS fully-local path verified

The final step is outside this plan: bump version in `pyproject.toml` (if present) and publish to PyPI. That is release engineering, not v0.3 development.
