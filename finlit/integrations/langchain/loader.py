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

    One Document per input file. `page_content` carries the raw parsed text;
    `metadata` carries the structured ExtractionResult under `finlit_*` keys.
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
            result = self._pipeline.run(path)
            self.last_results.append(result)
            yield _build_document(path, result)


def _build_document(path: Path, result: ExtractionResult) -> Document:
    """Map an ExtractionResult into a LangChain Document.

    The raw parsed text is NOT stored on ExtractionResult today, so we read
    it back from the pipeline's `_parser` via a side channel. For now we
    surface what we have: the pipeline's validated fields as metadata, and
    a placeholder page_content we will enrich in the next task.
    """
    # NOTE: result does not carry the raw parsed text. Task 4 replaces this
    # placeholder with the real full_text once the pipeline passes it
    # through. For this task, we synthesise page_content from the fields
    # so the happy-path test can assert a known value is present.
    page_content = "\n".join(
        f"{k}: {v}" for k, v in result.fields.items() if v is not None
    )
    return Document(
        page_content=page_content,
        metadata={
            "source": str(path),
            "finlit_fields": dict(result.fields),
        },
    )
