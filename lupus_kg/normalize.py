"""
Entity normalization and synonym resolution for the lupus knowledge graph.

Provides stable entity IDs via the pattern label::normalized_name so that
entities from different chunks MERGE into a single Neo4j node.
"""

from __future__ import annotations

import re


def load_synonyms(config: dict) -> dict[str, dict[str, str]]:
    """Build case-insensitive synonym lookup from YAML config.

    Returns: {label: {variant_lower: canonical_name}}
    """
    raw = config.get("synonyms") or {}
    result: dict[str, dict[str, str]] = {}
    for label, mappings in raw.items():
        if not isinstance(mappings, dict):
            continue
        result[label] = {k.lower(): v for k, v in mappings.items()}
    return result


def normalize_entity_name(
    name: str, label: str, synonyms: dict[str, dict[str, str]]
) -> str:
    """Resolve a name to its canonical form via the synonym table.

    Falls back to stripped original name if no synonym match.
    Collapses whitespace/newlines from PDF text extraction.
    """
    # Clean PDF artifacts: newlines, multiple spaces
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        return name
    label_syns = synonyms.get(label, {})
    canonical = label_syns.get(name.lower())
    if canonical:
        return canonical
    return name


def to_entity_id(label: str, name: str) -> str:
    """Build stable entity_id for MERGE: label::normalized_name.

    Same pattern as ~/public-company-graph/public_company_graph/kg_from_chunks.py.
    """
    norm = _normalize_for_id(name)
    return f"{label.lower()}::{norm}"


def _normalize_for_id(name: str) -> str:
    """Normalize a name for entity_id (lowercase, alphanumeric + underscores)."""
    if not name or not name.strip():
        return "unknown"
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_") or "unknown"
    return s[:200]
