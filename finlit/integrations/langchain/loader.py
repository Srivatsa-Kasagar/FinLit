"""FinLitLoader — LangChain BaseLoader wrapper around DocumentPipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from finlit.extractors.base import BaseExtractor
from finlit.integrations._schema_resolver import _resolve_schema
from finlit.pipeline import DocumentPipeline
from finlit.result import ExtractionResult
from finlit.schema import Schema


PathLike = Union[str, Path]


class FinLitLoader(BaseLoader):
    """Load files through a FinLit DocumentPipeline and emit LangChain Documents.

    One Document per input file. `page_content` is the raw parsed text from
    Docling; `metadata` carries the structured ExtractionResult under
    `finlit_*` keys (plus `source` per LangChain convention).

    Construction:
        FinLitLoader(path, schema="cra.t4")                 # build pipeline
        FinLitLoader(path, schema=my_schema)                # Schema instance
        FinLitLoader(path, pipeline=my_pipeline)            # inject pipeline
        FinLitLoader(path, pipeline=p, schema="cra.t4")     # pipeline wins
    """

    def __init__(
        self,
        file_path: PathLike | list[PathLike],
        *,
        schema: Schema | str | None = None,
        extractor: str | BaseExtractor = "claude",
        pipeline: DocumentPipeline | None = None,
    ) -> None:
        if isinstance(file_path, (str, Path)):
            self._paths: list[Path] = [Path(file_path)]
        else:
            self._paths = [Path(p) for p in file_path]

        if pipeline is not None:
            self._pipeline = pipeline
        elif schema is not None:
            self._pipeline = DocumentPipeline(
                schema=_resolve_schema(schema),
                extractor=extractor,
            )
        else:
            raise ValueError(
                "FinLitLoader requires either schema=... or pipeline=..."
            )

        self.last_results: list[ExtractionResult | None] = []

    def lazy_load(self) -> Iterator[Document]:
        self.last_results = []
        for path in self._paths:
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
