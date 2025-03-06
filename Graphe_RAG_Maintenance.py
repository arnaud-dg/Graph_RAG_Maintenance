import os, neo4j, asyncio, json
from dotenv import load_dotenv
from neo4j_graphrag.llm import OpenAILLM as LLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings 
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from tqdm import tqdm
import pandas as pd

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
    {"label": "Technicien", "description": "Le nom ou le visa d'un technicien", "properties": [{"name": "name", "type": "STRING"}]},
    {"label": "Action", "description": "Une activité déployée par une personne ou un service pour résoudre une panne", "properties": [{"name": "name", "type": "STRING"}, {"name": "date", "type": "DATE"}]},
    {"label": "Panne", "description": "Une panne ou un problème constaté sur une machine", "properties": [{"name": "name", "type": "STRING"}, {"name": "Durée", "type": "STRING"}, {"name": "Identifiant", "type": "STRING"}]},
    {"label": "Machine", "description": "Un équipement de production", "properties": [{"name": "name", "type": "STRING"}]},
    {"label": "Composant", "description": "Une pièce ou une partie d'une machine", "properties": [{"name": "name", "type": "STRING"}, {"name": "Référence", "type": "STRING"}]},
    {"label": "Cause", "description": "La cause ou root-cause ayant provoquée une panne", "properties": [{"name": "name", "type": "STRING"}]},
]

# Mise à jour des relations
relations = [
    # Relations de base
    "CONTIENT",
    "PROVOQUE",
    "REALISE",
    "INTERVIENT_SUR",
    "DIAGNOSTIQUE",
    "AFFECTE",
    "IMPLIQUE",    
    "CONTRIBUE_A",
    "DEGRADE"
]

prompt_template = '''
Vous êtes un technicien de maintenance dont la tâche est d'extraire des informations à partir de documents techniques et 
de rapports d'intervention de GMAO, puis de les structurer sous forme de graphe de propriétés afin de résoudre des problèmes.

Vous recevrez des rapports techniques concernant des pannes de machines, des défaillances et leurs causes racines. Votre mission sera :
- d'extraire les entités (nœuds) et de spécifier leur type,
- d'extraire les relations entre ces nœuds (la direction de la relation va du nœud de départ au nœud d'arrivée),
- d'inclure toutes les propriétés pertinentes pour chaque nœud en fonction du schéma fourni.

Informations sur le schéma :
{schema}

- Utilisez uniquement les informations provenant du texte d'entrée ci-dessous. N'ajoutez aucune information supplémentaire que vous pourriez avoir.
- Si le texte d'entrée est vide, retournez un JSON vide.
- Assurez-vous de créer autant de nœuds et de relations que nécessaire pour offrir un contexte médical riche en vue de recherches approfondies.
- Créez des nœuds avec toutes les propriétés disponibles du schéma lorsque l'information est présente dans le texte.
- Un assistant de connaissance basé sur l'IA doit pouvoir lire ce graphe et comprendre immédiatement le contexte pour formuler des questions de recherche détaillées.
- La propriété "name" est OBLIGATOIRE pour tous les nœuds. Si vous ne disposez pas de l'information exacte, utilisez une terminologie générique.

Ne retournez aucune autre information en dehors d'un JSON VALIDE.

RÈGLES DE FORMAT IMPORTANTES :
1. Retournez UNIQUEMENT un JSON valide - aucun autre texte avant ou après.
2. Toutes les chaînes de caractères doivent utiliser des guillemets doubles, et non des guillemets simples.
3. La réponse doit contenir à la fois les tableaux "nodes" et "relationships", même s'ils sont vides.
4. Les identifiants (ID) doivent être des chaînes de caractères et non des nombres (ex. : "0" et non 0).
5. Chaque nœud doit avoir un id, un label et toutes les propriétés définies dans le schéma (utilisez `null` si l'information n'est pas disponible).
6. Les résultats doivent être en français.

**Retournez strictement un JSON valide respectant ce format :**

{{
  "nodes": [
    {{
      "id": "0",
      "label": "Cause",
      "properties": {{
        "name": "Coupure électrique",
      }}
    }},
    {{
      "id": "1",
      "label": "Panne",
      "properties": {{
        "name":"disjonction du connecteur electrique",
        "Identifiant": "Cas_1",
        "Durée": "30 minutes"
      }}
    }}
  ],
  "relationships": [
    {{
      "type": "PROVOQUE",
      "start_node_id": "0",
      "end_node_id": "1",
      "properties": {{
        "details": "Coupure générale du réseau électrique"
      }}
    }}
  ]
}}

{examples}

Maintenant, réalise ta tâche en analysant le texte suivant :

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
    """Formate le schéma pour le prompt en incluant les descriptions et propriétés"""
    node_descriptions = []
    for node in nodes:
        properties_str = ", ".join([f"{prop['name']} ({prop['type'].lower()})" for prop in node['properties']])
        node_str = f"- {node['label']}: {node['description']}\n  Properties: {properties_str}"
        node_descriptions.append(node_str)

    return f"""Node Types:
{chr(10).join(node_descriptions)}

Relationship Types Available:
{', '.join(relations)}"""

kg_builder_csv = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    text_splitter=FixedSizeSplitter(chunk_size=1000, chunk_overlap=100),
    embedder=embedder,
    entities=nodes,
    relations=relations,
    prompt_template=prompt_template,
    from_pdf=False
)

async def process_json_file(csv_file_path):

    # Ouverture du fichier csv
    print(f"Processing csv file: {csv_file_path}")
    data = pd.read_csv(csv_file_path)
    print(data.head())

    # Transforme la colonne "Rapport d'Intervention" en liste
    interventions = data["Rapport d'Intervention"].tolist()
    
    # Création d'un fichier json  
    counter = 0
    for intervention in interventions:
        try:
            failure_id = "Case_" + str(counter)
            technicien = "Technician_" + data.loc[counter, 'Technicien']
            date = data.loc[counter, 'Date']
            replace_piece = replace_piece = str(data.loc[counter, 'Pièce Remplacée']) if not pd.isna(data.loc[counter, 'Pièce Remplacée']) else "None"
            full_text = failure_id + " - " + date + " - " + technicien + " - " + intervention + " - " + replace_piece
            print(f"full_text: {full_text}")
            
            try:
                # Utiliser le pipeline existant pour traiter le texte
                result = await kg_builder_csv.run_async(text=full_text)
                
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
        counter += 1

if __name__ == "__main__":
    csv_file_path = "data/Interventions_Presses_Fette.csv"
    asyncio.run(process_json_file(csv_file_path))