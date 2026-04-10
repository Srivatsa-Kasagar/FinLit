"""Shared pytest fixtures for FinLit tests."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from finlit.extractors.base import BaseExtractor
from finlit.parsers.docling_parser import ParsedDocument
from finlit.schema import Schema


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class _StubExtractionOutput:
    """Minimal duck-typed stand-in for ExtractionOutput."""

    def __init__(self, fields: dict[str, Any], confidence: dict[str, float], notes: str = ""):
        self.fields = fields
        self.confidence = confidence
        self.notes = notes


class StubExtractor(BaseExtractor):
    """Returns a pre-built ExtractionOutput-like object, no LLM calls."""

    def __init__(self, fields: dict[str, Any], confidence: dict[str, float], notes: str = ""):
        self._fields = fields
        self._confidence = confidence
        self._notes = notes

    def extract(self, text: str, schema: Schema) -> _StubExtractionOutput:
        return _StubExtractionOutput(self._fields, self._confidence, self._notes)


@pytest.fixture
def sample_t4_text() -> str:
    return (FIXTURES_DIR / "sample_t4.txt").read_text()


@pytest.fixture
def synthetic_parsed_document(sample_t4_text: str) -> ParsedDocument:
    return ParsedDocument(
        full_text=sample_t4_text,
        tables=[],
        metadata={
            "source": "/tmp/fake_t4.pdf",
            "sha256": "deadbeef" * 8,
            "filename": "fake_t4.pdf",
            "num_pages": 1,
        },
        source_path="/tmp/fake_t4.pdf",
    )


@pytest.fixture
def high_confidence_t4_extractor() -> StubExtractor:
    return StubExtractor(
        fields={
            "employer_name": "Acme Corp",
            "employee_sin": "123-456-789",
            "tax_year": 2024,
            "box_14_employment_income": 87500.00,
            "box_16_cpp_contributions": 3867.50,
            "box_18_ei_premiums": 1049.12,
            "box_22_income_tax_deducted": 15200.00,
            "box_24_ei_insurable_earnings": 61500.00,
            "box_26_cpp_pensionable_earnings": 68500.00,
            "box_44_union_dues": 0.0,
            "box_46_charitable_donations": 250.0,
            "box_50_rpp_or_dpsp_registration": None,
            "box_52_pension_adjustment": 0.0,
            "province_of_employment": "ON",
        },
        confidence={
            "employer_name": 0.99,
            "employee_sin": 0.99,
            "tax_year": 0.99,
            "box_14_employment_income": 0.98,
            "box_16_cpp_contributions": 0.97,
            "box_18_ei_premiums": 0.97,
            "box_22_income_tax_deducted": 0.97,
            "box_24_ei_insurable_earnings": 0.95,
            "box_26_cpp_pensionable_earnings": 0.95,
            "box_44_union_dues": 0.99,
            "box_46_charitable_donations": 0.95,
            "box_50_rpp_or_dpsp_registration": 0.0,
            "box_52_pension_adjustment": 0.99,
            "province_of_employment": 0.99,
        },
    )
