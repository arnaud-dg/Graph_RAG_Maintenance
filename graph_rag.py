import os, neo4j, asyncio, json
from dotenv import load_dotenv
from neo4j_graphrag.llm import OpenAILLM as LLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings 
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter

# Neo4j
load_dotenv()
neo4j_driver = neo4j.GraphDatabase.driver(
   os.getenv("NEO4J_URI"), 
   auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

llm = LLM(
   model_name="gpt-4o-mini",
   model_params={"response_format": {"type": "json_object"}, "temperature": 0}
)

embedder = OpenAIEmbeddings()

# text_splitter = FixedSizeSplitter(chunk_size=400, chunk_overlap=100)
basic_node_labels = ["Object", "Entity", "Group", "Person", "Organization", "Place"]
medical_node_labels = ["Anatomy", "Drug", "Symptom", "Disease", "Condition"]
node_labels = basic_node_labels + medical_node_labels

# define relationship types
rel_types = ["ACTIVATES", "AFFECTS", "ASSESSES", "ASSOCIATED_WITH", "AUTHORED",
    "BIOMARKER_FOR", "CAUSES", "CITES", "CONTRIBUTES_TO", "DESCRIBES", "EXPRESSES",
    "HAS_REACTION", "HAS_SYMPTOM", "INCLUDES", "INTERACTS_WITH", "PRESCRIBED",
    "PRODUCES", "RECEIVED", "RESULTS_IN", "TREATS", "USED_FOR"]

prompt_template = '''
You are a medical researcher whose task is to extract information from medical papers
and structuring it in a property graph to inform further medical and research Q&A.

You will be given medical texts about Alzheimer disease and you will:
- extract the entities (nodes) and specify their type
- extract the relationships between these nodes (the relationship direction goes from the start node to the end node)

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and
the relationship direction.

Use the following node labels and relationships:

basic_node_labels = ["Object", "Entity", "Group", "Person", "Organization", "Place"]

academic_node_labels = ["ArticleOrPaper", "PublicationOrJournal"]

medical_node_labels = ["Anatomy", "BiologicalProcess", "Cell", "CellularComponent",
                       "CellType", "Condition", "Disease", "Drug",
                       "EffectOrPhenotype", "Exposure", "GeneOrProtein", "Molecule",
                       "MolecularFunction", "Pathway"]


relationship types = ["ACTIVATES", "AFFECTS", "ASSESSES", "ASSOCIATED_WITH", "AUTHORED",
    "BIOMARKER_FOR", "CAUSES", "CITES", "CONTRIBUTES_TO", "DESCRIBES", "EXPRESSES",
    "HAS_REACTION", "HAS_SYMPTOM", "INCLUDES", "INTERACTS_WITH", "PRESCRIBED",
    "PRODUCES", "RECEIVED", "RESULTS_IN", "TREATS", "USED_FOR"]


- Use only the information from the Input text below.  Do not add any additional information you may have.
- If the input text is empty, return empty Json.
- Make sure to create as many nodes and relationships as needed to offer rich medical context for further research.
- An AI knowledge assistant must be able to read this graph and immediately understand the context to inform detailed research questions.
- Multiple documents will be ingested from different sources and we are using this property graph to connect information,
so make sure entity types are fairly general.

Do not return any additional information other than the VALID JSON in it.

IMPORTANT FORMAT RULES:
1. Return ONLY valid JSON - no other text before or after
2. All strings must use double quotes, not single quotes
3. The response must contain both "nodes" and "relationships" arrays, even if empty
4. IDs must be strings, not numbers (e.g., "0" not 0)
5. Every node must have id, label, and properties with a name
6. Every relationship must have type, start_node_id, end_node_id, and properties


**Strictly return valid JSON output following this format:**

{{
  "nodes": [
    {{
      "id": "0",
      "label": "EntityType",
      "properties": {{
        "name": "EntityName"
      }}
    }},
    {{
      "id": "1",
      "label": "AnotherEntityType",
      "properties": {{
        "name": "AnotherEntityName"
      }}
    }}
  ],
  "relationships": [
    {{
      "type": "TYPE_OF_RELATIONSHIP",
      "start_node_id": "0",
      "end_node_id": "1",
      "properties": {{
        "details": "Description of the relationship"
      }}
    }}
  ]
}}

Use only fhe following nodes and relationships (if provided):
{schema}

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and
the relationship direction.

Do not return any additional information other than the JSON in it.

Examples:
{examples}


Now, do your task. This is the Input text:

{text}

'''

kg_builder_pdf = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver ,
    text_splitter=FixedSizeSplitter(chunk_size=1000, chunk_overlap=100),
    embedder=embedder,
    entities=node_labels,
    relations=rel_types,
    prompt_template=prompt_template,
    from_pdf=True
)

pdfs_folder = "pdfs"

async def process_pdfs():
    pdf_file_paths = [os.path.join(pdfs_folder, file) for file in os.listdir(pdfs_folder) if file.endswith(".pdf")]
    
    for path in pdf_file_paths:
        print(f"Processing : {path}")
        pdf_result = await kg_builder_pdf.run_async(file_path=path)
        print(f"Result: {pdf_result}")
        await asyncio.sleep(2)  # Petite pause pour éviter de surcharger le traitement

# Exécution correcte de l'async avec asyncio.run()
if __name__ == "__main__":
    asyncio.run(process_pdfs())
