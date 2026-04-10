"""Tests for finlit.schema — Schema and Field dataclasses and YAML loading."""
from pathlib import Path
import pytest

from finlit.schema import Schema, Field


SCHEMAS_DIR = Path(__file__).parent.parent / "finlit" / "schemas"


def test_load_cra_t4_from_yaml():
    schema = Schema.from_yaml(SCHEMAS_DIR / "cra" / "t4.yaml")
    assert schema.name == "cra_t4"
    assert schema.version == "2024"
    assert "T4" in schema.document_type
    # T4 YAML defines 14 fields
    assert len(schema.fields) == 14


def test_t4_has_sin_field_with_pii_flag_and_regex():
    schema = Schema.from_yaml(SCHEMAS_DIR / "cra" / "t4.yaml")
    sin = schema.get_field("employee_sin")
    assert sin is not None
    assert sin.pii is True
    assert sin.required is True
    assert sin.regex == r"^\d{3}-\d{3}-\d{3}$"
    assert sin.dtype is str


def test_t4_box_14_is_float_and_required():
    schema = Schema.from_yaml(SCHEMAS_DIR / "cra" / "t4.yaml")
    box14 = schema.get_field("box_14_employment_income")
    assert box14 is not None
    assert box14.dtype is float
    assert box14.required is True


def test_field_names_returns_all_fields_in_order():
    schema = Schema.from_yaml(SCHEMAS_DIR / "cra" / "t4.yaml")
    names = schema.field_names()
    assert "employer_name" in names
    assert "employee_sin" in names
    assert len(names) == len(schema.fields)


def test_get_field_returns_none_for_unknown_field():
    schema = Schema.from_yaml(SCHEMAS_DIR / "cra" / "t4.yaml")
    assert schema.get_field("nonexistent_field") is None


def test_field_dataclass_defaults():
    f = Field(name="foo")
    assert f.dtype is str
    assert f.required is False
    assert f.pii is False
    assert f.regex is None
    assert f.description == ""
    assert f.aliases == []
