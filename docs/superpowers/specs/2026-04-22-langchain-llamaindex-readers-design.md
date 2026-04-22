# LangChain & LlamaIndex Reader Integrations — Design

**Status:** Draft
**Date:** 2026-04-22
**Owner:** Caseonix / FinLit maintainers
**Roadmap item:** `LangChain and LlamaIndex reader integrations` (README.md:454)

---

## 1. Goal

Expose FinLit's extraction pipeline as a native document loader in the two dominant Python LLM frameworks — LangChain and LlamaIndex — so developers building RAG pipelines, agents, or chains over Canadian financial documents can drop FinLit in with a single `pip` extra and one line of code, while preserving FinLit's structured fields, confidence scores, source references, PII entities, and audit log as queryable metadata on every emitted `Document`.

## 2. Non-goals

- A new extractor, parser, or schema. This feature is purely a presentation layer over the existing `DocumentPipeline`.
- Retrievers, vector stores, text splitters, or chain/agent classes. FinLit produces `Document` objects; downstream composition is the user's concern.
- Async API. `DocumentPipeline.run` is sync today; faking `async` with a thread pool would add surface without real concurrency benefit. Revisit when a real async path lands in core.
- A hosted/server-side integration. FinLit is a library (CLAUDE.md rule).
- Non-Canadian schemas or framework-specific document types.

## 3. Scope

Ship **LangChain first** as v0.1 of this feature. LlamaIndex is a planned follow-up that will mirror the same shape and share the schema-resolution helper. LlamaIndex is out of scope for this spec; a separate spec will cover it once LangChain has shipped and absorbed any real-world feedback.

## 4. Public API

### 4.1 Module location

```
finlit/integrations/langchain/
├── __init__.py    # exports FinLitLoader
└── loader.py      # FinLitLoader implementation
```

Import:

```python
from finlit.integrations.langchain import FinLitLoader
```

Core `finlit/__init__.py` is **not** modified. `finlit.integrations` remains an empty namespace package. This matches the CLAUDE.md rule that the top-level public API is frozen at `DocumentPipeline`, `BatchPipeline`, `Schema`, `Field`, `ExtractionResult`, `schemas`, `VisionExtractor`, `BaseVisionExtractor`.

### 4.2 Class signature

```python
class FinLitLoader(BaseLoader):
    def __init__(
        self,
        file_path: str | Path | list[str | Path],
        *,
        schema: str | Schema | None = None,
        extractor: str | BaseExtractor = "claude",
        pipeline: DocumentPipeline | None = None,
        on_error: Literal["raise", "skip", "include"] = "raise",
        include_audit_log: bool = False,
    ) -> None: ...

    def lazy_load(self) -> Iterator[Document]: ...
    # load() is inherited from BaseLoader as list(self.lazy_load())
```

### 4.3 Construction rules

- **Pipeline wins.** If `pipeline` is provided, `schema` and `extractor` are ignored. Otherwise the loader constructs `DocumentPipeline(schema=<resolved>, extractor=extractor)` once in `__init__` and reuses it across every path.
- **Schema resolution.** `schema` accepts three forms: a `Schema` object, a dotted registry key (`"cra.t4"`, `"banking.bank_statement"`), or the Python registry name (`"CRA_T4"`). Resolution lives in `_resolve_schema()` — a private helper kept separate so the future LlamaIndex reader can share it.
- **Fail fast.** If neither `schema` nor `pipeline` is provided, `__init__` raises `ValueError("Pass either schema=... or pipeline=...")`. No deferred failure at `load()` time.
- **Path normalization.** `file_path` is coerced to `list[Path]` in `__init__`. A single path becomes a one-element list. The `lazy_load` loop operates on one shape only.

### 4.4 Usage examples

```python
# One-liner (README example shape)
from finlit.integrations.langchain import FinLitLoader
docs = FinLitLoader("t4.pdf", schema="cra.t4").load()

# Batch
docs = FinLitLoader(
    ["t4_001.pdf", "t4_002.pdf", "t4_003.pdf"],
    schema="cra.t4",
    on_error="include",
).load()

# Power user — shared pipeline, custom extractor
pipeline = DocumentPipeline(schema=my_schema, extractor=my_custom_extractor)
loader = FinLitLoader("t4.pdf", pipeline=pipeline)
for doc in loader.lazy_load():
    vectorstore.add_documents([doc])
```

## 5. Document shape

One `Document` per input file. `page_content` carries the raw parsed text from Docling; structured extraction output lives in `metadata`.

### 5.1 Page content

`page_content = parsed.full_text` — the same text that was fed into the LLM extractor. This composes cleanly with LangChain text splitters for RAG, and keeps the "structured fields on top of raw text" story intact.

### 5.2 Metadata contract

```python
metadata = {
    # LangChain convention — top-level scalar, filterable in every vector store
    "source": str(path),

    # FinLit top-level scalars — filterable, cheap to store
    "finlit_schema": result.schema_name,                  # "cra.t4"
    "finlit_model": result.extractor_model,               # "anthropic:claude-sonnet-4-6"
    "finlit_extraction_path": result.extraction_path,     # "text" | "vision"
    "finlit_needs_review": result.needs_review,           # bool
    "finlit_extracted_field_count": result.extracted_field_count,

    # FinLit structured blobs — one level of nesting, tolerated by Chroma/Weaviate/pgvector
    "finlit_fields": result.fields,                       # dict[str, Any]
    "finlit_confidence": result.confidence,               # dict[str, float]
    "finlit_source_ref": result.source_ref,               # dict[str, dict]
    "finlit_warnings": result.warnings,                   # list[dict]
    "finlit_review_fields": result.review_fields,         # list[dict]
    "finlit_pii_entities": result.pii_entities,           # list[dict]

    # Opt-in only (omitted by default)
    # "finlit_audit_log": result.audit_log,
}
```

### 5.3 Why this exact shape

- `source` top-level matches community loader convention (`PyPDFLoader`, `WebBaseLoader`, etc.), so LangChain chains that assume `metadata["source"]` keep working.
- Every FinLit-owned key is namespaced with `finlit_` to prevent collisions when users merge Document lists from multiple loaders.
- Scalar keys at the top are filterable in every mainstream vector store. One level of dict/list nesting (`finlit_fields`, `finlit_confidence`) is the common ground that Chroma, Weaviate, Pinecone, and pgvector all tolerate.
- `finlit_audit_log` is gated behind `include_audit_log=False` because (a) it is often the largest blob in the result, (b) most RAG users don't need it in every vector row, (c) users who do need it can flip the flag.

### 5.4 Sidecar access

`loader.last_results: list[ExtractionResult]` is populated as extractions complete. This gives power users access to the original `ExtractionResult` objects (identical order to the yielded Documents) without having to reconstruct anything from metadata. Reset on each call to `lazy_load` / `load`.

## 6. Error handling

The `on_error` parameter controls per-path behavior inside the `lazy_load` loop:

| Mode | Behavior on failure |
|---|---|
| `"raise"` (default) | Re-raise original exception, abort iteration |
| `"skip"` | Call `logger.warning("FinLit extraction failed for %s: %s", path, exc)`, continue |
| `"include"` | Yield `Document(page_content="", metadata={"source": str(path), "finlit_error": repr(exc), "finlit_error_type": type(exc).__name__})`, continue |

No automatic retries. Retry logic is a caller concern.

**Why `"include"` exists:** in a PIPEDA/audit context, silently dropping files from a batch is a compliance hazard. `"include"` lets callers see every input they submitted, distinguish failures from successes, and write their own recovery loop. It pairs naturally with `finlit_needs_review` for downstream triage.

**Default `"raise"`:** keeps the single-file one-liner honest and matches Python norms. Power users running batches flip the flag.

## 7. Packaging

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "ruff", "mypy"]
langchain = ["langchain-core>=0.3.0"]
```

Install path: `pip install finlit[langchain]`.

**Why `langchain-core` and not `langchain`:** `BaseLoader` and `Document` live in `langchain-core`. Depending on the full `langchain` package drags in retrievers, chains, and community integrations we do not need, and forces our users into LangChain's release cadence on unrelated modules. `langchain-core` is the stable, narrow surface every LangChain user already has installed.

**Version pin rationale:** `langchain-core>=0.3.0` — the 0.3 line finalized the `BaseLoader.lazy_load` contract we rely on and is the minimum version supported by the current `langchain-community` ecosystem. No upper bound; we track their breaking changes on the normal open-source cadence.

**Import guard:** `loader.py` imports `langchain_core.documents.Document` and `langchain_core.document_loaders.BaseLoader` at module top. If `langchain-core` is not installed, Python's `ImportError` fires at `from finlit.integrations.langchain import FinLitLoader` time. We catch it in `finlit/integrations/langchain/__init__.py` and re-raise with:

```
ImportError("finlit[langchain] extras not installed. Run: pip install finlit[langchain]")
```

Core FinLit never imports `finlit.integrations`, so users who don't install the extra never pay a dependency cost and never see a broken import.

## 8. Testing

All tests run with no network. Real LLM calls are banned by CLAUDE.md. Tests inject a stub `BaseExtractor` subclass that returns a hardcoded `ExtractionOutput`, so the loader runs `DocumentPipeline` end-to-end over deterministic input without touching Anthropic/OpenAI/Ollama.

Test fixtures live in `tests/integrations/` and reuse `tests/fixtures/sample_t4.txt` where possible.

| Test | What it verifies |
|---|---|
| `test_single_file_load` | `FinLitLoader(path).load()` returns exactly one `Document`; every expected metadata key is present |
| `test_list_of_paths` | Three paths yield three Documents in input order |
| `test_pipeline_injection_overrides_kwargs` | Pre-built `pipeline` is used; `schema` and `extractor` kwargs ignored |
| `test_schema_resolution_forms` | `"cra.t4"`, `"CRA_T4"`, and a raw `Schema` instance all resolve correctly |
| `test_missing_schema_and_pipeline_raises` | `ValueError` at `__init__` time |
| `test_lazy_load_is_streaming` | `lazy_load()` returns a generator; items yielded one at a time; generator not exhausted by partial iteration |
| `test_on_error_raise` | One bad path in a list of three → exception, iteration aborts, no partial Documents yielded before failure |
| `test_on_error_skip` | Bad path logged and skipped; good ones yield normally |
| `test_on_error_include` | Bad path yields a Document with `page_content==""`, `metadata["finlit_error"]` and `metadata["finlit_error_type"]` set |
| `test_audit_log_default_omitted` | Default load: `"finlit_audit_log"` not in metadata |
| `test_audit_log_include_flag` | `include_audit_log=True` → `"finlit_audit_log"` present |
| `test_metadata_contract_snapshot` | Full metadata dict matches a pinned snapshot (locks the contract) |
| `test_last_results_sidecar` | `loader.last_results` contains `ExtractionResult` objects in input order after `load()` |
| `test_source_top_level` | `metadata["source"]` equals `str(path)` — LangChain convention held |
| `test_import_guard_message` | With `langchain-core` mocked absent via `sys.modules`, import raises the custom `ImportError` with the install hint |

## 9. Documentation updates

- Flip `- [ ] LangChain and LlamaIndex reader integrations` in `README.md:454` to a partial-completion split: `- [x] LangChain reader integration (v0.4)` and `- [ ] LlamaIndex reader integration`.
- Add a "LangChain integration" subsection to the README's Usage section with the one-liner and the sidecar `last_results` example.
- Add `examples/langchain_rag.py` — end-to-end: load T4 PDFs, split, embed with OpenAI, query a Chroma store, filter by `finlit_fields.box_14_employment_income > 50000`.

## 10. Out of scope / future work

- **LlamaIndex reader.** Will be a separate spec and PR. Reuses `_resolve_schema` and the metadata shape; substitutes `BaseReader.load_data` for `BaseLoader.lazy_load`, `Document` from `llama_index.core.schema`. Should ship within one or two releases of LangChain.
- **Directory loader.** `FinLitDirectoryLoader` for globbing directories. Defer until requested.
- **Parent-document retriever support.** Emitting one parent + N per-field child Documents. Defer until requested.
- **Async API.** Wait for `DocumentPipeline` to gain a real async entrypoint.
- **Integrations with LangGraph tools, LangChain tool-calling agents, or MCP.** MCP is already a separate roadmap item; the LangChain tool wrapper is a different shape than a loader.

## 11. Risks

- **LangChain API drift.** `langchain-core` minor versions have broken `BaseLoader` subclasses before. Mitigation: pin `>=0.3.0`, cover the loader with snapshot tests, watch CI on dependency updates.
- **Vector store metadata limits.** Some vector stores drop nested dicts silently. Mitigation: keep the most important signals (`finlit_needs_review`, `finlit_extraction_path`, etc.) at the top level as scalars; document the one-level-nesting compromise in the README.
- **Confusion with LocalMind.** FinLit's LangChain loader must not reach into LocalMind internals. Mitigation: already enforced by CLAUDE.md — FinLit remains standalone; LocalMind consumes FinLit, not the other way around.

## 12. Acceptance criteria

- `pip install -e ".[langchain]"` installs cleanly; `pip install -e "."` still works without LangChain.
- `from finlit.integrations.langchain import FinLitLoader` succeeds when `langchain-core` is installed; raises the custom `ImportError` when not.
- All 15 tests in §8 pass.
- `ruff check finlit/ tests/` clean.
- `mypy finlit/` clean.
- One-liner from §4.4 runs against a real T4 PDF with `ANTHROPIC_API_KEY` set and returns a `Document` whose metadata matches the §5.2 contract.
- README roadmap updated per §9.
