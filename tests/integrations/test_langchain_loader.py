"""Tests for finlit.integrations.langchain.FinLitLoader."""
from __future__ import annotations


def test_single_file_load_returns_one_document(
    t4_pipeline, patch_docling_parser, fake_t4_pdf
):
    from finlit.integrations.langchain import FinLitLoader

    loader = FinLitLoader(fake_t4_pdf, pipeline=t4_pipeline)
    docs = loader.load()

    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["source"] == str(fake_t4_pdf)
    assert doc.metadata["finlit_fields"]["employer_name"] == "Acme Corp"
    assert "Acme Corp" in doc.page_content  # raw parsed text surfaces through
