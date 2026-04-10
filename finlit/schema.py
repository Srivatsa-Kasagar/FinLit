"""
Schema and Field definitions for FinLit document extraction.

Schemas can be loaded from YAML files (built-in registry) or
constructed programmatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import yaml


@dataclass
class Field:
    """Defines a single extractable field in a document schema."""

    name: str
    dtype: Type = str
    required: bool = False
    pii: bool = False
    regex: str | None = None
    description: str = ""
    aliases: list[str] = field(default_factory=list)


@dataclass
class Schema:
    """Extraction schema for a document type."""

    name: str
    version: str = "1.0"
    document_type: str = ""
    description: str = ""
    fields: list[Field] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Schema":
        with open(path) as f:
            data = yaml.safe_load(f)
        fields = [
            Field(
                name=fd["name"],
                dtype=_parse_dtype(fd.get("dtype", "str")),
                required=fd.get("required", False),
                pii=fd.get("pii", False),
                regex=fd.get("regex"),
                description=fd.get("description", ""),
                aliases=fd.get("aliases", []),
            )
            for fd in data.get("fields", [])
        ]
        return cls(
            name=data["name"],
            version=str(data.get("version", "1.0")),
            document_type=data.get("document_type", ""),
            description=data.get("description", ""),
            fields=fields,
        )

    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    def get_field(self, name: str) -> Field | None:
        return next((f for f in self.fields if f.name == name), None)


def _parse_dtype(dtype_str: str) -> Type:
    return {"str": str, "float": float, "int": int, "bool": bool}.get(dtype_str, str)
