import os, neo4j, asyncio, json
from dotenv import load_dotenv
from neo4j_graphrag.llm import OpenAILLM as LLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings 
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from tqdm import tqdm

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

# Mise à jour des types de nœuds
nodes = [
    "Drug",
    "Trigger",
    "Patient",
    "Condition",
    "Gender",
    "Age",
    "Organization",
    "Place",
    "Disease",
    "Symptom",
    "Effect",
    "Disorder",
    "Treatment",
    "Route"
]

# Mise à jour des relations
relations = [
    # Relations de base
    "TRIGGERS",
    "HAS_EFFECT",
    "TREATS",
    "ADMINISTERS",
    "HAS_ATTRIBUTE",
    "CAUSES",
    
    # Relations composées
    "ADMINISTERED_TO",
    "ADMINISTERED_VIA",
    "COMBINED_WITH",
    
    # Relations générales existantes pertinentes
    "AFFECTS",
    "ASSOCIATED_WITH",
    "CONTRIBUTES_TO",
    "HAS_REACTION",
    "INTERACTS_WITH",
    "RESULTS_IN"
]

prompt_template = '''
You are a medical researcher whose task is to extract information from medical papers
and structuring it in a property graph to inform further medical and research Q&A.

You will be given medical texts about adverse effects and you will:
- extract the entities (nodes) and specify their type
- extract the relationships between these nodes (the relationship direction goes from the start node to the end node)

{schema}

- Use only the information from the Input text below. Do not add any additional information you may have.
- If the input text is empty, return empty Json.
- Make sure to create as many nodes and relationships as needed to offer rich medical context for further research.
- An AI knowledge assistant must be able to read this graph and immediately understand the context to inform detailed research questions.

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

{examples}

Now, do your task. This is the Input text:

{text}
'''

def check_prompt(text):
    """Vérifie que le prompt est correctement formaté"""
    test_context = {
        'schema': format_schema(),
        'examples': '',
        'text': text
    }
    
    try:
        formatted_prompt = prompt_template.format(**test_context)
        print("Prompt formatting successful")
        return formatted_prompt
    except Exception as e:
        print(f"Error formatting prompt: {e}")
        return None

def format_schema():
    return f"""Available node types:
{', '.join(nodes)}

Available relationship types:
{', '.join(relations)}"""

kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    text_splitter=FixedSizeSplitter(chunk_size=1000, chunk_overlap=100),
    embedder=embedder,
    entities=nodes,
    relations=relations,
    prompt_template=prompt_template,
    from_pdf=False
)

async def process_json_file(json_file_path):
    print(f"Processing JSON file: {json_file_path}")
    
    # Test du prompt avec la première entrée
    with open(json_file_path, 'r', encoding='utf-8') as f:
        test_entry = json.loads(next(f))
        print("\nTesting prompt with first entry...")
        test_prompt = check_prompt(test_entry['context'])
        if test_prompt:
            print("Sample formatted prompt (first 500 chars):")
            print(test_prompt[:500])
    
    counter = 0
    with open(json_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            counter += 1
            # si counter inférieur à 1000
            if counter < 1000:
                if line.strip():  # Ignorer les lignes vides
                    try:
                        data = json.loads(line)
                        context = data['context']
                        # print(f"Context: {context}")
                        
                        try:
                            # Utiliser le pipeline existant pour traiter le texte
                            result = await kg_builder.run_async(text=context)
                            
                            # Vérifier si le résultat est une chaîne de caractères
                            if isinstance(result, str):
                                # print("Raw result from LLM:")
                                # print(result)
                                try:
                                    parsed_result = json.loads(result)
                                    print("Parsed result:")
                                    print(json.dumps(parsed_result, indent=2))
                                except json.JSONDecodeError as je:
                                    print(f"Failed to parse LLM response as JSON: {je}")
                                    print("First 500 characters of response:", result[:500])
                            else:
                                print(f"Raw JSON line: {counter}")
                                print(f"\nProcessing entry ID: {data['id']}")
                                # print(f"Unexpected result type: {type(result)}")
                                print("Result content:", result)
                        
                        except Exception as e:
                            print(f"Error during pipeline processing: {str(e)}")
                            print(f"Error type: {type(e)}")
                            import traceback
                            print("Traceback:")
                            print(traceback.format_exc())
                        
                        await asyncio.sleep(1)
                        
                    except json.JSONDecodeError as e:
                        print(f"Error decoding input JSON line: {e}")
                    except Exception as e:
                        print(f"Error processing entry: {e}")
                        import traceback
                        print(traceback.format_exc())
            else:
                break

if __name__ == "__main__":
    json_file_path = "data/Phee_dataset.json"
    asyncio.run(process_json_file(json_file_path))