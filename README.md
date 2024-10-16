# GraphRAG Python package Examples

This repository contains a worked example using the [GraphRAG Python package](https://neo4j.com/docs/neo4j-graphrag-python/current/index.html) for Neo4j. The example demonstrates the end-to-end workflow, starting from unstructured documents (in this case pdfs), to knowledge graph construction, knowledge graph retriever design, and a working GraphRAG pipeline. Research papers on Lupus are used as the data source. 

1. The [end-to-end-lupus](end-to-end-lupus.ipynb) notebook contains the worked example.
2. The [corresponding blog post](https://neo4j.com/graphrag-python-package/) has a full write-up walking through the example with more details, explanations, and resources. 
3. The [truncated-pdfs](truncated-pdfs) directory contains the pdf source files.  They were obtained from [NIH PubMed](https://pubmed.ncbi.nlm.nih.gov/). Some pages at the end containing references have been truncated to better focus the knowledge graph on medical information rather than citations and other publications. 


