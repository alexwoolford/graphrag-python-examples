from openai import OpenAI
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings


from dotenv import load_dotenv
import os

os.environ["OPENAI_API_BASE"] = "http://10.0.1.37:8001/v1"
os.environ["OPENAI_API_KEY"] = "dummy"

EMBEDDING_BASE  = "http://10.0.1.37:7997"
EMBEDDING_MODEL = "bge"

# load neo4j credentials (and openai api key in background).
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')


LLM_BASE_URL = "http://10.0.1.37:8001/v1"   # vLLM server
EMBEDDING_BASE_URL  = "http://10.0.1.37:7997"      # Infinity server
EMBEDDING_MODEL = "bge"

def make_clients():
    # --- Embeddings ---------------------------------------------------------
    embeddings = OpenAIEmbeddings(
        EMBEDDING_MODEL,
        base_url=EMBEDDING_BASE,   # http://10.0.1.37:7997
        api_key="local",           # dummy key; server ignores it
        timeout=30.0,
    )

    # --- LLM ---------------------------------------------------------------
    llm = OpenAILLM(
        "qwen2",
        base_url=os.environ["OPENAI_API_BASE"],
        api_key=os.environ["OPENAI_API_KEY"],
        model_params={
            "temperature": 0.0,
            "max_tokens": 1024,
        },
        timeout=60.0
    )
    return llm, embeddings

import neo4j
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
driver        = neo4j.GraphDatabase.driver(NEO4J_URI,
                                              auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
                                              database=NEO4J_DATABASE)
llm, embedder = make_clients()
splitter      = FixedSizeSplitter(1000, 200)


NODE_TYPES = [
    "Bill", "Title", "Subtitle", "Section", "Agency", "Program",
    "Requirement", "Appropriation", "BeneficiaryGroup",
    "Deadline", "Act"
]

RELATIONSHIP_TYPES = [
    "HAS_TITLE", "HAS_SUBTITLE", "HAS_SECTION", "ESTABLISHES",
    "AUTHORIZES_FUNDS", "APPROPRIATES_TO", "IMPOSES_REQUIREMENT",
    "BENEFITS", "AMENDS", "ADMINISTERED_BY", "AFFECTS",
    "APPLIES_TO", "EFFECTIVE_ON"
]

PATTERNS = [
    ("Bill", "HAS_TITLE", "Title"),
    ("Title", "HAS_SUBTITLE", "Subtitle"),
    ("Subtitle", "HAS_SECTION", "Section"),
    ("Section", "HAS_SECTION", "Section"),          # nested §§
    ("Section", "ESTABLISHES", "Program"),
    ("Section", "AUTHORIZES_FUNDS", "Appropriation"),
    ("Section", "IMPOSES_REQUIREMENT", "Requirement"),
    ("Section", "BENEFITS", "BeneficiaryGroup"),
    ("Section", "AMENDS", "Act"),
    ("Program", "ADMINISTERED_BY", "Agency"),
    ("Program", "BENEFITS", "BeneficiaryGroup"),
    ("Appropriation", "APPROPRIATES_TO", "Program"),
    ("Requirement", "AFFECTS", "Agency"),
    ("Requirement", "APPLIES_TO", "BeneficiaryGroup"),
    ("Section", "EFFECTIVE_ON", "Deadline"),
]


def build_schema_section() -> str:
    """Yield a markdown bulleted list that the LLM will see."""
    nodes = "\n".join(f"* **{n}**" for n in NODE_TYPES)
    rels  = "\n".join(f"* `{r}`"  for r in RELATIONSHIP_TYPES)
    return f"""
**Node labels**

{nodes}

**Relationship types**

{rels}
""".strip()

prompt_template = """
You are a legislative analyst tasked with extracting structured information from U.S. federal bills and representing it as a property graph that will power a GraphRAG question-answering system.

**Task**

1. **Extract every entity** (node) the *Input text* clearly mentions and assign
   it one of the allowed labels below.
2. **Extract every relationship** that exists *within* the *Input text*, using
   the correct direction (from the `start_node_id` to the `end_node_id`).

**Output format**

Return **only** a single JSON object in the form

{{{{
  "nodes": [
    {{{{ "id": "0", "label": "<NodeLabel>", "properties": {{{{ "name": "<entity name>" }}}} }}}},
    ...
  ],
  "relationships": [
    {{{{ "type": "<REL_TYPE>", "start_node_id": "0", "end_node_id": "1", "properties": {{{{ "details": "<optional short description>" }}}} }}}},
    ...
  ]
}}}}

* Each `id` must be unique *within this chunk*.
* Include only properties that appear literally in the text.
* If the input is empty return `{{}}`.

**Allowed node labels and relationship types**

{schema}

**Guidelines**

* Do **not** invent information not present in the text.
* Use the containment edges (`HAS_TITLE`, `HAS_SUBTITLE`, `HAS_SECTION`).
* Use policy-specific edges (`APPROPRIATES_TO`, `EFFECTIVE_ON`, …) only when
  the text supports them.
* Keep entity types general so they can be reused across bills.

**Few-shot examples**

{examples}

---

**Input text**

{text}
"""

prompt_ready = prompt_template.format(
    schema=build_schema_section(),
    examples="",
    text="{text}"
)

from neo4j_graphrag.generation.prompts import ERExtractionTemplate

json_example = '''
{
  "nodes":[
    {"id":"0","label":"Bill","properties":{"name":"H.R. 1"}},
    {"id":"1","label":"Title","properties":{"name":"Big Beautiful Bill"}},
    {"id":"2","label":"Section","properties":{"name":"SEC. 2001"}},
    {"id":"3","label":"Agency","properties":{"name":"Secretary of Agriculture"}},
    {"id":"4","label":"Program","properties":{"name":"Farm Tech Grant Program"}}
  ],
  "relationships":[
    {"type":"HAS_TITLE","start_node_id":"0","end_node_id":"1","properties":{}},
    {"type":"HAS_SECTION","start_node_id":"1","end_node_id":"2","properties":{}},
    {"type":"ESTABLISHES","start_node_id":"2","end_node_id":"4","properties":{"details":"shall establish"}},
    {"type":"ADMINISTERED_BY","start_node_id":"4","end_node_id":"3","properties":{}}
  ]
}
'''

esc = json_example.replace('{', '{{').replace('}', '}}')

raw_prompt = """
**Task**
1. Extract every entity in the text and label it one of:
   {node_types}

2. Extract every relationship in the text and choose one of:
   {rel_types}

**Allowed patterns** (direction matters):
{patterns}

**Example Input Text:**
SEC. 2001. The Secretary of Agriculture shall establish the Farm Tech Grant Program.

**Example Output:**
{json_ex}

**Guidelines:**
- Extract all entities mentioned in the text, even if they seem minor
- Create relationships between entities based on the text's meaning
- Use the most specific relationship type that matches the text
- Include details in relationship properties when the text provides them
- Maintain the hierarchical structure (Bill -> Title -> Section -> etc.)

Now process this text:
{{text}}
"""

prompt_str = raw_prompt.format(
    node_types=NODE_TYPES,
    rel_types=RELATIONSHIP_TYPES,
    patterns=PATTERNS,
    json_ex=esc
)

clean_template = ERExtractionTemplate(prompt_str)

from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    text_splitter=FixedSizeSplitter(chunk_size=1000, chunk_overlap=200),
    embedder=embedder,
    entities=NODE_TYPES,
    relations=RELATIONSHIP_TYPES,
    potential_schema=PATTERNS,
    # enforce_schema="STRICT",
    prompt_template=clean_template,
    from_pdf=True,
    # on_error="RAISE"
)

pdf_file_paths = ['BILLS-119hr1rh-sample.pdf']

async def process_pdfs():
    for path in pdf_file_paths:
        print(f"Processing : {path}")
        result = await kg_builder.run_async(file_path=path)
        print(f"Result: {result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(process_pdfs())




