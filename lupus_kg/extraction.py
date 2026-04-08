"""
Extract a knowledge graph from Chunk nodes using GLiNER2.

Reads Chunk nodes from Neo4j, runs entity and relation extraction with
GLiNER2, normalizes entities via synonym table, enforces relationship
directions via RELATION_SCHEMA, and writes the results back to Neo4j
with provenance.

Not a CLI entry point — use scripts/extract_lupus_kg.py.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from lupus_kg.normalize import load_synonyms, normalize_entity_name, to_entity_id
from lupus_kg.schema import RELATION_SCHEMA, snake_to_pascal, validate_and_orient_relation

logger = logging.getLogger(__name__)


def _load_config(config_path: str | Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _create_constraints(session, labels: list[str]):
    """Create uniqueness constraints and name indexes for each entity label."""
    for label in labels:
        pascal = snake_to_pascal(label)
        try:
            session.run(
                f"CREATE CONSTRAINT unique_{label}_entity_id IF NOT EXISTS "
                f"FOR (n:{pascal}) REQUIRE n.entity_id IS UNIQUE"
            )
        except Exception as e:
            logger.debug("Constraint for %s: %s", pascal, e)
        try:
            session.run(
                f"CREATE INDEX idx_{label}_name IF NOT EXISTS "
                f"FOR (n:{pascal}) ON (n.name)"
            )
        except Exception as e:
            logger.debug("Index for %s: %s", pascal, e)


def _chunk_text_to_str(text_val) -> str:
    """Convert Chunk.text (may be list of strings) to a single string."""
    if isinstance(text_val, list):
        return " ".join(str(t) for t in text_val)
    return str(text_val or "")


def run(
    *,
    config_path: str | Path,
    limit: int | None = None,
    batch_size: int = 16,
    dry_run: bool = False,
    replace: bool = False,
    log: logging.Logger | None = None,
) -> dict[str, int]:
    """Extract KG from Chunk nodes using GLiNER2.

    Args:
        config_path: Path to YAML config.
        limit: Max chunks to process (None = all).
        batch_size: Batch size for extraction and writes.
        dry_run: If True, extract but do not write to Neo4j.
        replace: If True, re-extract chunks even if kg_extracted_at is set.
        log: Logger.

    Returns:
        Dict with processed, nodes_created, relationships_created, errors.
    """
    log = log or logger
    config = _load_config(config_path)

    # Load config values
    neo4j_cfg = config.get("neo4j") or {}
    database = neo4j_cfg.get("database")
    model_id = config.get("model_id", "fastino/gliner2-base-v1")
    entity_labels = (config.get("entities") or {}).get("labels", [])
    relation_labels = (config.get("relations") or {}).get("labels", [])
    pipeline_cfg = config.get("pipeline") or {}
    batch_size = pipeline_cfg.get("batch_size", batch_size)
    entity_threshold = pipeline_cfg.get("entity_threshold", 0.5)
    relation_threshold = pipeline_cfg.get("relation_threshold", 0.5)
    synonyms = load_synonyms(config)

    # Load Neo4j credentials from .env
    import os
    from dotenv import load_dotenv
    load_dotenv(".env", override=True)
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")
    database = database or os.getenv("NEO4J_DATABASE")

    # Load GLiNER2
    import torch
    from gliner2 import GLiNER2

    log.info("Loading GLiNER2 model %s...", model_id)
    extractor = GLiNER2.from_pretrained(model_id)
    if torch.cuda.is_available():
        extractor = extractor.to("cuda")
        log.info("Model on CUDA")
    else:
        log.info("Model on CPU")
    extractor.eval()

    # Connect to Neo4j
    import neo4j as neo4j_driver_mod
    driver = neo4j_driver_mod.GraphDatabase.driver(
        neo4j_uri, auth=(neo4j_user, neo4j_password)
    )

    # Create constraints
    with driver.session(database=database) as session:
        _create_constraints(session, entity_labels)

    # Read chunks
    where_kg = " AND c.kg_extracted_at IS NULL" if not replace else ""
    read_query = f"""
    MATCH (c:Chunk)
    WHERE c.text IS NOT NULL {where_kg}
    RETURN elementId(c) AS eid, c.text AS text
    """
    if limit is not None:
        read_query += f" LIMIT {limit}"

    log.info("Reading chunks from Neo4j...")
    with driver.session(database=database) as session:
        records = list(session.run(read_query))
    log.info("Found %d chunks to process", len(records))

    # Process in batches
    stats = {"processed": 0, "nodes_created": 0, "relationships_created": 0, "errors": 0}
    now = datetime.now(UTC).isoformat()
    extraction_model = model_id.split("/")[-1]

    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start : batch_start + batch_size]
        texts = [_chunk_text_to_str(r["text"]) for r in batch]
        eids = [r["eid"] for r in batch]

        # GLiNER2 extraction
        t0 = time.perf_counter()
        try:
            entity_results = extractor.batch_extract_entities(
                texts, entity_labels, threshold=entity_threshold
            )
            relation_results = extractor.batch_extract_relations(
                texts, relation_labels, threshold=relation_threshold
            )
        except Exception as e:
            log.error("Batch extraction failed: %s", e)
            stats["errors"] += len(batch)
            continue
        t1 = time.perf_counter()
        log.info(
            "Batch %d-%d: extracted in %.1fs",
            batch_start, batch_start + len(batch), t1 - t0,
        )

        if dry_run:
            for i, (ent, rel) in enumerate(zip(entity_results, relation_results)):
                log.info("Chunk %s entities: %s", eids[i][-8:], ent.get("entities", {}))
                log.info("Chunk %s relations: %s", eids[i][-8:], rel.get("relation_extraction", {}))
            stats["processed"] += len(batch)
            continue

        # Build nodes and relationships for this batch
        nodes_to_merge: list[dict] = []  # {label, entity_id, name, chunk_eid}
        rels_to_create: list[dict] = []

        for i, (ent_result, rel_result) in enumerate(zip(entity_results, relation_results)):
            chunk_eid = eids[i]
            text = texts[i]
            entities_map = ent_result.get("entities") or {}
            rel_map = rel_result.get("relation_extraction") or {}

            # Track entities extracted from this chunk for relation label lookup
            name_to_label: dict[str, str] = {}

            for label, names in entities_map.items():
                if not isinstance(names, list):
                    continue
                pascal = snake_to_pascal(label)
                for name in names:
                    if not name or not isinstance(name, str) or len(name.strip()) < 2:
                        continue
                    canonical = normalize_entity_name(name, label, synonyms)
                    eid = to_entity_id(label, canonical)
                    nodes_to_merge.append({
                        "label": pascal,
                        "entity_id": eid,
                        "name": canonical[:500],
                        "chunk_eid": chunk_eid,
                    })
                    name_to_label[name.strip().lower()] = pascal
                    name_to_label[canonical.lower()] = pascal

            # Process relations
            for rel_type, pairs in rel_map.items():
                rel_type = str(rel_type or "").strip().upper()
                if rel_type not in RELATION_SCHEMA:
                    continue
                for pair in (pairs if isinstance(pairs, list) else []):
                    if not isinstance(pair, (tuple, list)) or len(pair) < 2:
                        continue
                    head_name = str(pair[0]).strip()
                    tail_name = str(pair[1]).strip()
                    if not head_name or not tail_name:
                        continue

                    # Look up labels from extracted entities
                    head_label = name_to_label.get(head_name.lower())
                    tail_label = name_to_label.get(tail_name.lower())
                    if not head_label or not tail_label:
                        continue

                    # Validate direction and swap if needed
                    oriented = validate_and_orient_relation(
                        rel_type, head_name, head_label, tail_name, tail_label
                    )
                    if oriented is None:
                        continue

                    h_name, h_label, t_name, t_label = oriented
                    h_canonical = normalize_entity_name(h_name, h_label.lower(), synonyms)
                    t_canonical = normalize_entity_name(t_name, t_label.lower(), synonyms)
                    # Map PascalCase back to snake for entity_id
                    h_label_snake = re.sub(r"(?<!^)(?=[A-Z])", "_", h_label).lower()
                    t_label_snake = re.sub(r"(?<!^)(?=[A-Z])", "_", t_label).lower()
                    h_eid = to_entity_id(h_label_snake, h_canonical)
                    t_eid = to_entity_id(t_label_snake, t_canonical)

                    if h_eid == t_eid:
                        continue  # self-loop

                    rels_to_create.append({
                        "rel_type": rel_type,
                        "head_label": h_label,
                        "head_id": h_eid,
                        "head_name": h_canonical,
                        "tail_label": t_label,
                        "tail_id": t_eid,
                        "tail_name": t_canonical,
                        "chunk_eid": chunk_eid,
                        "evidence_text": text[:1000],
                    })

        # Dedupe nodes by entity_id
        seen_nodes: set[str] = set()
        unique_nodes: list[dict] = []
        for n in nodes_to_merge:
            if n["entity_id"] not in seen_nodes:
                seen_nodes.add(n["entity_id"])
                unique_nodes.append(n)

        # Write to Neo4j
        with driver.session(database=database) as session:
            # MERGE entity nodes grouped by label
            nodes_by_label: dict[str, list[dict]] = defaultdict(list)
            for n in unique_nodes:
                nodes_by_label[n["label"]].append(n)

            for label, rows in nodes_by_label.items():
                try:
                    result = session.run(
                        f"""
                        UNWIND $rows AS row
                        MERGE (n:{label} {{entity_id: row.entity_id}})
                        SET n.name = row.name, n.loaded_at = datetime()
                        """,
                        rows=[{"entity_id": r["entity_id"], "name": r["name"]} for r in rows],
                    )
                    stats["nodes_created"] += result.consume().counters.nodes_created
                except Exception as e:
                    log.warning("Node MERGE failed for %s: %s", label, e)
                    stats["errors"] += len(rows)

            # CREATE FROM_CHUNK edges (entity -> chunk) for retriever compatibility
            from_chunk_rows: list[dict] = []
            for n in nodes_to_merge:
                from_chunk_rows.append({
                    "entity_id": n["entity_id"],
                    "label": n["label"],
                    "chunk_eid": n["chunk_eid"],
                })

            # Group by label for efficient MATCH
            fc_by_label: dict[str, list[dict]] = defaultdict(list)
            for fc in from_chunk_rows:
                fc_by_label[fc["label"]].append(fc)

            for label, rows in fc_by_label.items():
                try:
                    session.run(
                        f"""
                        UNWIND $rows AS row
                        MATCH (e:{label} {{entity_id: row.entity_id}})
                        MATCH (c:Chunk) WHERE elementId(c) = row.chunk_eid
                        MERGE (e)-[:FROM_CHUNK]->(c)
                        """,
                        rows=[{"entity_id": r["entity_id"], "chunk_eid": r["chunk_eid"]} for r in rows],
                    )
                except Exception as e:
                    log.warning("FROM_CHUNK edge failed for %s: %s", label, e)

            # MERGE domain relationships with provenance
            rels_by_type: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
            for r in rels_to_create:
                key = (r["rel_type"], r["head_label"], r["tail_label"])
                rels_by_type[key].append(r)

            for (rel_type, head_label, tail_label), group in rels_by_type.items():
                try:
                    result = session.run(
                        f"""
                        UNWIND $rels AS r
                        MATCH (head:{head_label} {{entity_id: r.head_id}})
                        MATCH (tail:{tail_label} {{entity_id: r.tail_id}})
                        MERGE (head)-[rel:{rel_type} {{chunk_eid: r.chunk_eid}}]->(tail)
                        ON CREATE SET
                          rel.evidence_text = r.evidence_text,
                          rel.extracted_at = datetime(),
                          rel.extraction_method = 'gliner2_kg',
                          rel.extraction_model = $model
                        """,
                        rels=[{
                            "head_id": r["head_id"],
                            "tail_id": r["tail_id"],
                            "chunk_eid": r["chunk_eid"],
                            "evidence_text": r["evidence_text"],
                        } for r in group],
                        model=extraction_model,
                    )
                    stats["relationships_created"] += result.consume().counters.relationships_created
                except Exception as e:
                    log.warning("Relationship MERGE failed %s: %s", rel_type, e)
                    stats["errors"] += len(group)

            # Mark chunks as extracted
            chunk_eids = eids
            try:
                session.run(
                    """
                    UNWIND $eids AS eid
                    MATCH (c:Chunk) WHERE elementId(c) = eid
                    SET c.kg_extracted_at = datetime()
                    """,
                    eids=chunk_eids,
                )
            except Exception as e:
                log.warning("Chunk timestamp update failed: %s", e)

        stats["processed"] += len(batch)
        log.info(
            "Progress: %d/%d chunks, %d nodes, %d rels",
            stats["processed"], len(records),
            stats["nodes_created"], stats["relationships_created"],
        )

    driver.close()
    return stats
