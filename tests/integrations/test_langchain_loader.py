"""Tests for finlit.integrations.langchain.FinLitLoader."""
from __future__ import annotations

import inspect


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


def test_list_of_paths_preserves_order(
    t4_pipeline, patch_docling_parser, tmp_path
):
    from finlit.integrations.langchain import FinLitLoader

    p1 = tmp_path / "a.pdf"; p1.write_bytes(b"x")
    p2 = tmp_path / "b.pdf"; p2.write_bytes(b"x")
    p3 = tmp_path / "c.pdf"; p3.write_bytes(b"x")

    loader = FinLitLoader([p1, p2, p3], pipeline=t4_pipeline)
    docs = loader.load()

    assert [d.metadata["source"] for d in docs] == [str(p1), str(p2), str(p3)]


def test_lazy_load_is_a_generator(
    t4_pipeline, patch_docling_parser, fake_t4_pdf
):
    from finlit.integrations.langchain import FinLitLoader

    loader = FinLitLoader(fake_t4_pdf, pipeline=t4_pipeline)
    iterator = loader.lazy_load()

    assert inspect.isgenerator(iterator)
    # Pulling one item must not require iterating the whole list
    first = next(iterator)
    assert first.metadata["source"] == str(fake_t4_pdf)
