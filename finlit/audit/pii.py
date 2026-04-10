"""
PII detection and redaction using Microsoft Presidio.

Canadian recognizers registered on init:
  - CA_SIN: XXX-XXX-XXX (high confidence) or 9 unformatted digits (low confidence)
  - CA_POSTAL_CODE: A1A 1A1
  - CA_CRA_BN: 9-digit BN + RT/RP/RC/RZ account number
"""
from __future__ import annotations

from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


@dataclass
class PIIResult:
    original_text: str
    redacted_text: str
    detected_entities: list[dict]


class CanadianPIIDetector:
    """Presidio wrapper with custom Canadian recognizers."""

    def __init__(self) -> None:
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._register_canadian_recognizers()

    def _register_canadian_recognizers(self) -> None:
        sin = PatternRecognizer(
            supported_entity="CA_SIN",
            patterns=[
                Pattern(name="CA_SIN_formatted",
                        regex=r"\b\d{3}-\d{3}-\d{3}\b", score=0.9),
                Pattern(name="CA_SIN_unformatted",
                        regex=r"\b[0-9]{9}\b", score=0.4),
            ],
            context=["sin", "social insurance", "numéro d'assurance"],
        )
        postal = PatternRecognizer(
            supported_entity="CA_POSTAL_CODE",
            patterns=[Pattern(name="CA_POSTAL",
                              regex=r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b", score=0.85)],
        )
        cra_bn = PatternRecognizer(
            supported_entity="CA_CRA_BN",
            patterns=[Pattern(name="CA_BN",
                              regex=r"\b\d{9}\s?(?:RT|RP|RC|RZ)\d{4}\b",
                              score=0.9)],
            context=["business number", "bn", "account number"],
        )
        self._analyzer.registry.add_recognizer(sin)
        self._analyzer.registry.add_recognizer(postal)
        self._analyzer.registry.add_recognizer(cra_bn)

    def analyze(self, text: str, language: str = "en") -> list[dict]:
        results = self._analyzer.analyze(text=text, language=language)
        return [
            {
                "entity_type": r.entity_type,
                "score": round(r.score, 3),
                "start": r.start,
                "end": r.end,
                "text": text[r.start:r.end],
            }
            for r in results
        ]

    def redact(self, text: str, language: str = "en") -> PIIResult:
        analyzer_results = self._analyzer.analyze(text=text, language=language)
        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators={
                "CA_SIN": OperatorConfig("replace", {"new_value": "***-***-***"}),
                "CA_POSTAL_CODE": OperatorConfig("replace", {"new_value": "[POSTAL]"}),
                "CA_CRA_BN": OperatorConfig("replace", {"new_value": "[CRA-BN]"}),
                "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
            },
        )
        return PIIResult(
            original_text=text,
            redacted_text=anonymized.text,
            detected_entities=[
                {
                    "entity_type": r.entity_type,
                    "score": round(r.score, 3),
                    "start": r.start,
                    "end": r.end,
                }
                for r in analyzer_results
            ],
        )
