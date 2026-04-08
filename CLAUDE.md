# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational example demonstrating end-to-end GraphRAG using the Neo4j GraphRAG Python package and GLiNER2 for knowledge graph extraction. Processes Lupus research PDFs into a Neo4j knowledge graph, then performs retrieval-augmented question answering.

Blog walkthrough: https://neo4j.com/blog/graphrag-python-package/
Docs: https://neo4j.com/docs/neo4j-graphrag-python/current/index.html

## Environment Setup

```bash
conda env create -f environment.yml
conda activate graphrag-python-examples
cp .env.template .env   # then fill in NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE, OPENAI_API_KEY
jupyter lab
```

Requires: Python 3.11, active Neo4j instance, valid OpenAI API key (for embeddings and RAG answers).

## Architecture

The notebook `end-to-end-lupus.ipynb` runs cells sequentially through four phases:

1. **PDF Chunking & Embedding** — `SimpleKGPipeline` parses PDFs in `truncated-pdfs/`, splits text into chunks, and creates `Chunk`/`Document` nodes with OpenAI vector embeddings. Entity/relation lists are empty — no LLM-based extraction.
2. **KG Extraction with GLiNER2** — `lupus_kg.extraction.run()` reads Chunk nodes, extracts entities and relationships using the GLiNER2 model (`fastino/gliner2-base-v1`), normalizes via synonym table, and writes to Neo4j with schema-enforced relationship directions. Configured via `configs/lupus_kg.yaml`.
3. **Retrieval** — Vector index on Chunk embeddings. `VectorRetriever` (semantic only) and `VectorCypherRetriever` (semantic + graph traversal via `FROM_CHUNK` edges to entity neighborhood).
4. **GraphRAG QA** — `GraphRAG` pipeline combines retriever context with GPT-4o to answer questions grounded in the knowledge graph.

## Key Modules

- `configs/lupus_kg.yaml` — Entity/relation types, synonym table, GLiNER2 model config
- `lupus_kg/schema.py` — `RELATION_SCHEMA` dict enforcing correct edge directions (e.g., Drug→Disease for TREATS)
- `lupus_kg/normalize.py` — Synonym resolution and stable `entity_id` computation for MERGE-based dedup
- `lupus_kg/extraction.py` — Core GLiNER2 extraction pipeline
- `scripts/extract_lupus_kg.py` — CLI: `python scripts/extract_lupus_kg.py --config configs/lupus_kg.yaml`

## Key Dependencies

- `gliner2` — Local entity and relationship extraction (no API needed)
- `neo4j_graphrag[openai]` — Chunking, retrieval, and RAG pipeline
- `neo4j` — Database driver
- `pypdf` — PDF text extraction
- `python-dotenv` — Loads `.env` credentials

## Data

Four truncated Lupus research PDFs from NIH PubMed in `truncated-pdfs/`. Reference pages removed to focus on medical content.
