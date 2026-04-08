# GraphRAG Python Package: Demo Talk-Track

## Opening (2 min)

> "Today we're going to show you how the GraphRAG Python package and GLiNER2 turn unstructured medical research papers into a queryable knowledge graph — and then uses that graph to power more accurate, explainable GenAI answers. We built this from three Lupus research PDFs. No manual data modeling. No hand-crafted ETL. Just PDFs in, knowledge graph out."

---

## Section 1: What's in the Graph? (3 min)

> "Let's start by looking at what the pipeline built for us automatically."

### Query 1: Graph Overview — What did we extract?

```cypher
// What did GLiNER2 extract from 3 research PDFs?
MATCH (n)
WHERE NOT n:Chunk AND NOT n:Document
WITH labels(n) AS nodeLabels
UNWIND nodeLabels AS label
WHERE NOT label IN ['__KGBuilder__']
WITH label, count(*) AS total
RETURN label AS EntityType, total AS Count
ORDER BY total DESC
LIMIT 10
```

**Talking point:** *"From just 3 truncated PDFs, GLiNER2 extracted over 1,100 deduplicated entities — diseases, drugs, genes and proteins, biomarkers, symptoms, cell types, and more. Every relationship has schema-enforced direction: drugs always point to diseases via TREATS, never the reverse. This is the knowledge graph that powers our RAG."*

### Query 2: The Disease at the Center

```cypher
// SLE and its direct connections — the hub of our knowledge graph
MATCH (sle:Disease {name: 'Systemic Lupus Erythematosus'})-[r]->(target)
WHERE NOT type(r) IN ['FROM_CHUNK', 'NEXT_CHUNK', 'FROM_DOCUMENT']
RETURN type(r) AS Relationship, target.name AS ConnectedEntity, labels(target)[0] AS EntityType
ORDER BY Relationship, ConnectedEntity
```

**Talking point:** *"Systemic Lupus Erythematosus sits at the center of this graph — and notice it's one canonical node, not 13 duplicates like you'd get with LLM-based extraction. It's connected to symptoms, complications, and immunological processes. This structure is what gives GraphRAG its edge over plain vector search."*

---

## Section 2: Clinical Insights from Graph Traversal (5 min)

> "Now let's ask the kind of questions a researcher or clinician would actually ask."

### Query 3: What Treats Lupus?

```cypher
// What drugs treat lupus-related conditions? (direction enforced: Drug -> Disease)
MATCH (drug:Drug)-[:TREATS]->(disease:Disease)
RETURN drug.name AS Treatment, collect(DISTINCT disease.name) AS Conditions
ORDER BY Treatment
```

**Talking point:** *"Every TREATS relationship has the correct direction: Drug treats Disease. Hydroxychloroquine treats Systemic Lupus Erythematosus. Cyclophosphamide treats SLE. Rituximab treats autoimmunity. This is guaranteed by the RELATION_SCHEMA — it's structurally impossible to create a reversed relationship."*

### Query 4: Biomarkers for Diagnosing and Monitoring SLE

```cypher
// What biomarkers are linked to SLE diagnosis and monitoring?
MATCH (biomarker:Biomarker)-[:BIOMARKER_FOR]->(disease:Disease)
RETURN biomarker.name AS Biomarker, disease.name AS Disease
ORDER BY Biomarker
```

**Talking point:** *"Biomarkers are critical in lupus management. Here we see anti-dsDNA antibodies, complement C3, anti-Sm antibodies, epigenetic markers, and more — all correctly typed as Biomarker nodes pointing to Disease nodes. In a traditional RAG setup, these would be scattered across text chunks with no guaranteed connection."*

### Query 5: The Complexity of SLE — Symptoms

```cypher
// SLE affects multiple organ systems — what symptoms are documented?
MATCH (disease:Disease)-[:HAS_SYMPTOM]->(symptom:Symptom)
RETURN disease.name AS Disease, symptom.name AS Symptom
ORDER BY disease.name, symptom.name
```

**Talking point:** *"The graph captures the heterogeneity of lupus — pain, proteinuria, immunological abnormalities, organ involvement. Each symptom is typed and connected with the correct direction: Disease has Symptom."*

---

## Section 3: Multi-Hop Reasoning (5 min)

> "This is where knowledge graphs really shine — questions that require connecting multiple pieces of information."

### Query 6: Two-Hop Paths from SLE

```cypher
// Follow two relationships out from SLE to discover connected knowledge
MATCH (sle:Disease {name: 'Systemic Lupus Erythematosus'})-[r1]->(mid)-[r2]->(end)
WHERE NOT type(r1) IN ['FROM_CHUNK', 'NEXT_CHUNK', 'FROM_DOCUMENT']
  AND NOT type(r2) IN ['FROM_CHUNK', 'NEXT_CHUNK', 'FROM_DOCUMENT']
RETURN type(r1) AS Hop1, mid.name AS Through, labels(mid)[0] AS ThroughType,
       type(r2) AS Hop2, end.name AS Destination, labels(end)[0] AS DestType
```

**Talking point:** *"Here we traverse two hops from SLE. This multi-hop reasoning connects dots across entities that may appear in entirely different papers — that's the power of a knowledge graph."*

### Query 7: Causal Chains in Lupus

```cypher
// Causal chains extracted from the literature
MATCH (cause)-[:CAUSES]->(effect)
RETURN cause.name AS Cause, labels(cause)[0] AS CauseType,
       effect.name AS Effect, labels(effect)[0] AS EffectType
ORDER BY Cause
```

**Talking point:** *"The graph captures causal relationships extracted automatically from the literature. These causal chains give the LLM real reasoning paths when answering questions."*

---

## Section 4: The Power of Context in RAG (5 min)

> "Now let's see how this graph context transforms RAG quality."

### Query 8: Drug Neighborhood — How Graph Traversal Enriches Retrieval

```cypher
// Graph context around a drug entity
MATCH (drug:Drug {name: 'Rituximab'})-[r]-(connected)
WHERE NOT type(r) IN ['FROM_CHUNK', 'NEXT_CHUNK', 'FROM_DOCUMENT']
RETURN drug.name AS Drug, type(r) AS Relationship,
       startNode(r) = drug AS Outgoing,
       connected.name AS ConnectedEntity, labels(connected)[0] AS EntityType
```

**Talking point:** *"This is the kind of context the VectorCypherRetriever adds beyond raw text. Take Rituximab — the graph tells us what it treats, what it's associated with, and where it's referenced. When the LLM gets this structured context alongside the text chunk, the answers are consistently richer."*

### Query 9: Genes and Proteins Associated with SLE

```cypher
// What genetic/molecular factors are linked to SLE?
MATCH (gene:GeneOrProtein)-[:ASSOCIATED_WITH]->(disease:Disease)
WHERE disease.name CONTAINS 'Lupus' OR disease.name CONTAINS 'SLE'
RETURN gene.name AS GeneOrProtein, disease.name AS Disease
ORDER BY gene.name
```

**Talking point:** *"The graph connects molecular-level knowledge to disease-level understanding. HLA, complement, transcription factors — all linked to SLE through typed, directed relationships that the LLM can traverse for richer answers."*

---

## Section 5: Provenance and Traceability (2 min)

### Query 10: What Papers Feed This Knowledge Graph?

```cypher
// The provenance trail: which papers contributed what knowledge?
MATCH (entity)-[:FROM_CHUNK]->(chunk:Chunk)-[:FROM_DOCUMENT]->(doc:Document)
WHERE NOT entity:Chunk AND NOT entity:Document
WITH doc, count(DISTINCT entity) AS entitiesExtracted,
     count(DISTINCT chunk) AS chunksProcessed
RETURN doc.path AS Paper,
       chunksProcessed AS TextChunks,
       entitiesExtracted AS EntitiesExtracted
ORDER BY entitiesExtracted DESC
```

**Talking point:** *"Every piece of knowledge in this graph is traceable back to a source document and text chunk. This is explainability and governance built in — you can always ask 'where did this answer come from?' and trace it back to the original research paper. That's a critical requirement for regulated industries like healthcare."*

---

## Closing (1 min)

> "What you just saw was built with a YAML config and a few dozen lines of Python code. Three PDFs went in. A knowledge graph with over 1,100 deduplicated entities and schema-enforced relationships came out. GLiNER2 handles extraction locally — no API cost, fully deterministic. And that graph powers RAG that's more accurate, more explainable, and more useful than vector search alone."

**Resources:**
- Demo code: `github.com/neo4j-product-examples/graphrag-python-examples`
- Docs: `neo4j.com/docs/neo4j-graphrag-python`
- Blog: `neo4j.com/blog/graphrag-python-package/`
- Free course: `graphacademy.neo4j.com/courses/genai-workshop-graphrag/`
