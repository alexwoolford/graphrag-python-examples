"""
Relationship schema for the lupus knowledge graph.

RELATION_SCHEMA enforces correct edge directions: for each relation type,
it declares the valid (source_label, target_label) pair. Extraction code
uses this to swap reversed pairs or drop type-mismatched ones.
"""

from __future__ import annotations

import re

# Maps relation type -> (source_label, target_label) in PascalCase.
RELATION_SCHEMA: dict[str, tuple[str, str]] = {
    "TREATS":          ("Drug", "Disease"),
    "BIOMARKER_FOR":   ("Biomarker", "Disease"),
    "CAUSES":          ("Disease", "Symptom"),
    "ASSOCIATED_WITH": ("GeneOrProtein", "Disease"),
    "TARGETS":         ("Drug", "GeneOrProtein"),
    "INHIBITS":        ("Drug", "Pathway"),
    "EXPRESSED_IN":    ("GeneOrProtein", "Anatomy"),
    "HAS_SYMPTOM":     ("Disease", "Symptom"),
    "INTERACTS_WITH":  ("GeneOrProtein", "GeneOrProtein"),
    "PRODUCED_BY":     ("GeneOrProtein", "CellType"),
}


def snake_to_pascal(label: str) -> str:
    """Convert snake_case GLiNER2 label to PascalCase Neo4j label.

    Examples: gene_or_protein -> GeneOrProtein, disease -> Disease
    """
    return "".join(word.capitalize() for word in label.split("_"))


def validate_and_orient_relation(
    rel_type: str,
    head_name: str,
    head_label: str,
    tail_name: str,
    tail_label: str,
) -> tuple[str, str, str, str] | None:
    """Check if extracted relation matches RELATION_SCHEMA.

    Returns (head_name, head_label, tail_name, tail_label) with possible swap
    to correct orientation, or None if the relation should be dropped.
    """
    expected = RELATION_SCHEMA.get(rel_type)
    if expected is None:
        return None
    src_label, tgt_label = expected

    # Already correct
    if head_label == src_label and tail_label == tgt_label:
        return (head_name, head_label, tail_name, tail_label)

    # Reversed — swap
    if head_label == tgt_label and tail_label == src_label:
        return (tail_name, tail_label, head_name, head_label)

    # Symmetric relations (e.g., INTERACTS_WITH)
    if src_label == tgt_label and head_label == src_label:
        return (head_name, head_label, tail_name, tail_label)

    # Label mismatch — drop
    return None
