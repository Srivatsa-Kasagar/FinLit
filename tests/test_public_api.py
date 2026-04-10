"""Pins the public import surface of finlit."""
import finlit


def test_top_level_exports_present():
    assert hasattr(finlit, "DocumentPipeline")
    assert hasattr(finlit, "BatchPipeline")
    assert hasattr(finlit, "Schema")
    assert hasattr(finlit, "Field")
    assert hasattr(finlit, "ExtractionResult")
    assert hasattr(finlit, "schemas")


def test_schemas_registry_has_expected_names():
    from finlit import schemas
    assert schemas.CRA_T4.name == "cra_t4"
    assert schemas.CRA_T5.name == "cra_t5"
    assert schemas.CRA_T4A.name == "cra_t4a"
    assert schemas.CRA_NR4.name == "cra_nr4"
    assert schemas.BANK_STATEMENT.name == "bank_statement"


def test_version_string_present():
    assert isinstance(finlit.__version__, str)


def test_all_exports_restricted():
    assert set(finlit.__all__) == {
        "DocumentPipeline",
        "BatchPipeline",
        "Schema",
        "Field",
        "ExtractionResult",
        "schemas",
    }
