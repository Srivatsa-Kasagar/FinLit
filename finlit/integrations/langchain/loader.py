"""FinLitLoader — LangChain BaseLoader wrapper around DocumentPipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from finlit.pipeline import DocumentPipeline
from finlit.result import ExtractionResult


PathLike = Union[str, Path]


class FinLitLoader(BaseLoader):
    """Load files through a FinLit DocumentPipeline and emit LangChain Documents.

    One Document per input file. `page_content` is the raw parsed text from
    Docling; `metadata` carries the structured ExtractionResult under
    `finlit_*` keys (plus `source` per LangChain convention).
    """

    def __init__(
        self,
        file_path: PathLike | list[PathLike],
        *,
        pipeline: DocumentPipeline,
    ) -> None:
        if isinstance(file_path, (str, Path)):
            self._paths: list[Path] = [Path(file_path)]
        else:
            self._paths = [Path(p) for p in file_path]
        self._pipeline = pipeline
        self.last_results: list[ExtractionResult | None] = []

    def lazy_load(self) -> Iterator[Document]:
        self.last_results = []
        for path in self._paths:
            # Parse once for page_content. This is the same parser the
            # pipeline will use internally; Docling caches nothing stateful
            # across parse() calls, so the double-parse cost is acceptable
            # for v0.1. A future optimisation can thread the parsed text
            # through ExtractionResult to avoid the second call.
            parsed = self._pipeline._parser.parse(path)
            result = self._pipeline.run(path)
            self.last_results.append(result)
            yield _build_document(path, parsed.full_text, result)


def _build_document(
    path: Path, full_text: str, result: ExtractionResult
) -> Document:
    return Document(
        page_content=full_text,
        metadata={
            "source": str(path),
            "finlit_schema": result.schema_name,
            "finlit_model": result.extractor_model,
            "finlit_extraction_path": result.extraction_path,
            "finlit_needs_review": result.needs_review,
            "finlit_extracted_field_count": result.extracted_field_count,
            "finlit_fields": dict(result.fields),
            "finlit_confidence": dict(result.confidence),
            "finlit_source_ref": dict(result.source_ref),
            "finlit_warnings": list(result.warnings),
            "finlit_review_fields": list(result.review_fields),
            "finlit_pii_entities": list(result.pii_entities),
        },
    )
