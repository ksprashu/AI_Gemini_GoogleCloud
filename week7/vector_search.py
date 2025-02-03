# This will query the vector database (firestore) for the user query
# and return the closest matching documents for the user query

import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure


import re
import os
import json
import dotenv
import logging

# fetch project_id and location from environment
dotenv.load_dotenv()
project_id = os.environ.get("PROJECT_ID")
location = os.environ.get("REGION")
database_id = os.environ.get("DATABASE_ID")

# init vertex ai with project_id and location
vertexai.init(project=project_id, location=location)
model = TextEmbeddingModel.from_pretrained("text-embedding-004")

db = firestore.Client(project=project_id, database=database_id)
collection = db.collection("city-guides")

def _get_query_embedding(query):
    """get the embedding for the query"""
    query_input = TextEmbeddingInput(
                    text=query,
                    task_type="RETRIEVAL_QUERY"
                )
    result = model.get_embeddings([query_input])
    logging.info(f"got embedding of size {len(result[0].values)} for query")
    return result[0].values
    

def find_matching_docs(query):
    """do a nearest neighbour search on firestore on the vectors"""
    query_embedding = _get_query_embedding(query)

    vector_query = collection.find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_embedding),
        distance_measure=DistanceMeasure.COSINE,
        limit=5,
    )

    docs = list(vector_query.stream())
    logging.info(f"found {len(docs)} matching docs")
    for doc in docs:
        logging.info(f"title: {doc.to_dict()['title']}")
    return docs


