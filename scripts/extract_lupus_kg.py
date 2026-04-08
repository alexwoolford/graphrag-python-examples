#!/usr/bin/env python3
"""
Extract a knowledge graph from Chunk nodes using GLiNER2.

Reads Chunk nodes from Neo4j, runs entity and relation extraction,
then writes entity nodes and relationships back with provenance.

Usage:
  python scripts/extract_lupus_kg.py --config configs/lupus_kg.yaml
  python scripts/extract_lupus_kg.py --config configs/lupus_kg.yaml --limit 50 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from lupus_kg.extraction import run as run_kg_extraction


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract KG from chunk nodes (GLiNER2 entities + relations -> Neo4j)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config path (entity labels, relation labels, model_id, synonyms)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max chunks to process (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for GLiNER2 and writes (default: 16)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run extraction but do not write to Neo4j",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Re-process chunks even if kg_extracted_at is set",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("extract_lupus_kg")

    try:
        stats = run_kg_extraction(
            config_path=args.config,
            limit=args.limit,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            replace=args.replace,
            log=log,
        )
        log.info(
            "Done: processed=%d nodes_created=%d relationships_created=%d errors=%d",
            stats["processed"],
            stats["nodes_created"],
            stats["relationships_created"],
            stats["errors"],
        )
        return 0 if stats["errors"] == 0 else 1
    except Exception as e:
        log.exception("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
