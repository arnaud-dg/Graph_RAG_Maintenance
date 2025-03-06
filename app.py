import os
import neo4j
import asyncio
import json
import streamlit as st
from dotenv import load_dotenv
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import VectorRetriever

# Charger les variables d'environnement
load_dotenv()

# Connexion à Neo4j
neo4j_driver = neo4j.GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# Configuration du modèle de langage
llm = OpenAILLM(
    model_name="gpt-4o-mini",
    model_params={"temperature": 0}
)

# Configuration de l'embedder avec la bonne dimension
embedder = OpenAIEmbeddings(model="text-embedding-3-large")
INDEX_NAME = "my_vector_index"  # Assurez-vous que le nom correspond à l'index créé dans Neo4j

# Initialisation du retriever avec la bonne dimension
retriever = VectorRetriever(driver=neo4j_driver, index_name=INDEX_NAME, embedder=embedder)

# Initialisation de GraphRAG
rag = GraphRAG(retriever=retriever, llm=llm)

async def query_graph(question: str):
    """
    Fonction qui prend une question en langage naturel et retourne une réponse basée sur le graphe Neo4j
    """
    try:
        # Exécuter la requête avec GraphRAG
        response = rag.search(query_text=question, retriever_config={"top_k": 5})
        return response.answer
    except Exception as e:
        return f"Erreur lors de l'interrogation du graphe : {str(e)}"

# Interface Streamlit
st.title("Chatbot GraphRAG pour Neo4j")
st.write("Posez vos questions en langage naturel pour interroger la base de données Neo4j.")

# Zone d'entrée utilisateur
question = st.text_input("Votre question :", "Quelles sont les principales causes des pannes des machines Fette ?")

if st.button("Poser la question"):
    if question:
        with st.spinner("Interrogation en cours..."):
            response = asyncio.run(query_graph(question))
            st.write(response)
