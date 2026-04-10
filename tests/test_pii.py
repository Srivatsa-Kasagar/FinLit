"""Tests for finlit.audit.pii.CanadianPIIDetector."""
import pytest

from finlit.audit.pii import CanadianPIIDetector, PIIResult


@pytest.fixture(scope="module")
def detector() -> CanadianPIIDetector:
    return CanadianPIIDetector()


def test_detects_sin_in_text(detector: CanadianPIIDetector):
    results = detector.analyze("Employee SIN: 123-456-789 on file.")
    kinds = {r["entity_type"] for r in results}
    assert "CA_SIN" in kinds


def test_detects_canadian_postal_code(detector: CanadianPIIDetector):
    results = detector.analyze("Mailing address M5V 3A8 Toronto")
    kinds = {r["entity_type"] for r in results}
    assert "CA_POSTAL_CODE" in kinds


def test_detects_cra_business_number(detector: CanadianPIIDetector):
    results = detector.analyze("Business Number 123456789RT0001")
    kinds = {r["entity_type"] for r in results}
    assert "CA_CRA_BN" in kinds


def test_redact_replaces_sin_with_placeholder(detector: CanadianPIIDetector):
    result = detector.redact("SIN on file: 123-456-789")
    assert isinstance(result, PIIResult)
    assert "123-456-789" not in result.redacted_text
    assert "***-***-***" in result.redacted_text


def test_redact_populates_detected_entities(detector: CanadianPIIDetector):
    result = detector.redact("SIN 123-456-789 postal M5V 3A8")
    assert len(result.detected_entities) >= 2
