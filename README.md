# GraphRAG Python Package Examples

This repository demonstrates an end-to-end [GraphRAG](https://neo4j.com/docs/neo4j-graphrag-python/current/index.html) workflow using Neo4j, with [GLiNER2](https://github.com/fastino/gliner2) for knowledge graph extraction. Starting from unstructured Lupus research PDFs, it builds a knowledge graph with schema-enforced relationship directions, then uses that graph to power retrieval-augmented question answering.

> **Note:** The [original blog post](https://neo4j.com/blog/graphrag-python-package/) describes an earlier version that used LLM-based entity extraction via `SimpleKGPipeline`. This repository has since replaced that approach with GLiNER2 for higher precision, correct relationship directions, and deterministic local extraction. The retrieval and GraphRAG components remain the same.

## How It Works

The [end-to-end-lupus](end-to-end-lupus.ipynb) notebook runs through four phases:

1. **PDF Chunking & Embedding** — `SimpleKGPipeline` (from `neo4j_graphrag`) parses PDFs, splits text into chunks, and creates `Chunk`/`Document` nodes with OpenAI vector embeddings. Entity/relation lists are empty — no LLM-based extraction.
2. **KG Extraction with GLiNER2** — The `lupus_kg` module extracts entities and relationships from chunks using GLiNER2 (`fastino/gliner2-base-v1`). A `RELATION_SCHEMA` enforces correct directions (e.g., Drug→Disease for TREATS). A synonym table merges variants (SLE, lupus → Systemic Lupus Erythematosus) into canonical entities.
3. **Retrieval** — `VectorRetriever` (semantic search) and `VectorCypherRetriever` (semantic + graph traversal) retrieve relevant context from the knowledge graph.
4. **GraphRAG** — The `GraphRAG` class combines retriever context with GPT-4o to answer natural language questions grounded in the knowledge graph.

## Prerequisites

- Python 3.11
- A Neo4j instance with the [APOC plugin](https://neo4j.com/docs/apoc/current/installation/) installed
- An OpenAI API key (for embeddings and RAG answer generation — entity extraction runs locally via GLiNER2)

## Setup

```bash
conda env create -f environment.yml
conda activate graphrag-python-examples
cp .env.template .env
```

Edit `.env` with your credentials:

```
NEO4J_URI=neo4j+s://<instance>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=<database>
OPENAI_API_KEY=sk-...
```

Then open the notebook:

```bash
jupyter lab
# Open end-to-end-lupus.ipynb and run cells sequentially
```

## Project Structure

- `end-to-end-lupus.ipynb` — Main notebook with the full workflow
- `configs/lupus_kg.yaml` — Entity/relation types, synonym table, GLiNER2 model config
- `lupus_kg/` — GLiNER2 extraction pipeline (schema enforcement, normalization, Neo4j writes)
- `scripts/extract_lupus_kg.py` — CLI alternative: `python scripts/extract_lupus_kg.py --config configs/lupus_kg.yaml`
- `truncated-pdfs/` — Source PDFs from [NIH PubMed](https://pubmed.ncbi.nlm.nih.gov/), with reference pages removed

## Resources

- [GraphRAG Python Package Docs](https://neo4j.com/docs/neo4j-graphrag-python/current/index.html)
- [Original Blog Post](https://neo4j.com/blog/graphrag-python-package/) (describes the earlier LLM-based approach)
- [Free GraphRAG Course](https://graphacademy.neo4j.com/courses/genai-workshop-graphrag/)
