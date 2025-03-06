import os
import neo4j
import asyncio
import json
import streamlit as st
from dotenv import load_dotenv
from neo4j_graphrag.llm import OpenAILLM as LLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.query import GraphRAG

# Charger les variables d'environnement
load_dotenv()

# Connexion à Neo4j
neo4j_driver = neo4j.GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# Configuration du modèle de langage
llm = LLM(
    model_name="gpt-4o-mini",
    model_params={"response_format": {"type": "json_object"}, "temperature": 0}
)

embedder = OpenAIEmbeddings()

# Initialisation de GraphRAG
graph_rag = GraphRAG(
    llm=llm,
    driver=neo4j_driver,
    embedder=embedder,
)

async def query_graph(question: str):
    """
    Fonction qui prend une question en langage naturel et retourne une réponse basée sur le graphe Neo4j
    """
    try:
        print(f"Question utilisateur : {question}")
        
        # Effectuer la requête GraphRAG
        response = await graph_rag.run_async(question=question)
        
        # Vérifier si la réponse est une chaîne JSON valide
        if isinstance(response, str):
            try:
                parsed_response = json.loads(response)
                return parsed_response
            except json.JSONDecodeError as je:
                print(f"Erreur de parsing JSON : {je}")
                print("Premiers 500 caractères de la réponse :", response[:500])
        else:
            return response
    except Exception as e:
        print(f"Erreur lors de l'interrogation du graphe : {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

# Interface Streamlit
st.title("Chatbot GraphRAG pour Neo4j")
st.write("Posez vos questions en langage naturel pour interroger la base de données Neo4j.")

# Zone d'entrée utilisateur
question = st.text_input("Votre question :", "Quelles sont les principales causes des pannes des machines Fette ?")

if st.button("Poser la question"):
    if question:
        with st.spinner("Interrogation en cours..."):
            response = asyncio.run(query_graph(question))
            
            if response:
                st.json(response)
            else:
                st.error("Une erreur s'est produite lors de la requête.")
