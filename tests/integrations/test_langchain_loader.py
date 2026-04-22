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


def test_metadata_contract_snapshot(
    t4_pipeline, patch_docling_parser, fake_t4_pdf
):
    """Lock the metadata schema. Adding a new field is OK (update the test);
    changing a field name is a breaking change this test is designed to
    catch before it merges."""
    from finlit.integrations.langchain import FinLitLoader

    loader = FinLitLoader(fake_t4_pdf, pipeline=t4_pipeline)
    doc = loader.load()[0]

    # page_content is the raw parsed text, not a synthesized summary
    assert "T4" in doc.page_content
    assert "Acme Corp" in doc.page_content
    assert "87500.00" in doc.page_content

    expected_keys = {
        "source",
        "finlit_schema",
        "finlit_model",
        "finlit_extraction_path",
        "finlit_needs_review",
        "finlit_extracted_field_count",
        "finlit_fields",
        "finlit_confidence",
        "finlit_source_ref",
        "finlit_warnings",
        "finlit_review_fields",
        "finlit_pii_entities",
    }
    assert set(doc.metadata.keys()) == expected_keys, (
        f"metadata drift: {set(doc.metadata.keys()) ^ expected_keys}"
    )

    assert doc.metadata["finlit_schema"] == "cra_t4"
    assert doc.metadata["finlit_extraction_path"] == "text"
    assert doc.metadata["finlit_needs_review"] is False
    assert isinstance(doc.metadata["finlit_fields"], dict)
    assert isinstance(doc.metadata["finlit_confidence"], dict)
