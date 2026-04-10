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


def test_default_analyze_excludes_us_only_entities(detector: CanadianPIIDetector):
    """By default analyze() should suppress US-only Presidio recognizers
    that generate noise on Canadian documents (e.g. US_DRIVER_LICENSE
    firing on 'T4' on every T4 slip we tested)."""
    # Presidio's US_DRIVER_LICENSE recognizer reliably fires on the literal
    # string "T4" — we observed this on every real T4 slip during testing.
    text = "T4 Statement of Remuneration Paid."
    results = detector.analyze(text)
    kinds = {r["entity_type"] for r in results}
    assert "US_DRIVER_LICENSE" not in kinds
    assert "US_SSN" not in kinds
    assert "US_PASSPORT" not in kinds
    assert "US_BANK_NUMBER" not in kinds
    assert "US_ITIN" not in kinds


def test_exclude_entities_override_allows_us_entities(
    detector: CanadianPIIDetector,
):
    """Callers can pass an empty set to get Presidio's raw output including
    US-only recognizers."""
    text = "T4 Statement of Remuneration Paid."
    results = detector.analyze(text, exclude_entities=set())
    kinds = {r["entity_type"] for r in results}
    # With no exclusions, US_DRIVER_LICENSE should fire on 'T4'
    assert "US_DRIVER_LICENSE" in kinds
