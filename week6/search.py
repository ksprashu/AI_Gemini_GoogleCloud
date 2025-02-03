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


# Encapsulate all this logic in a Search class
class Search():
    """This class will handle all the database interactions"""

    def __init__(self):
        """This will initialise the firestore client"""
        logging.info(f"initialising firestore client and collections")

        # fetch project_id and location from environment
        dotenv.load_dotenv()
        project_id = os.environ.get("PROJECT_ID")
        location = os.environ.get("REGION")
        database_id = os.environ.get("DATABASE_ID")

        # init vertex ai with project_id and location
        vertexai.init(project=project_id, location=location)
        self._model = TextEmbeddingModel.from_pretrained("text-embedding-004")

        db = firestore.Client(project=project_id, database=database_id)
        self._guides = db.collection("city-guides")
        self._user = db.collection("user-history")

        def _get_query_embedding(query):
            """get the embedding for the query"""
            query_input = TextEmbeddingInput(
                            text=query,
                            task_type="RETRIEVAL_QUERY"
                        )
            result = self._model.get_embeddings([query_input])
            logging.info(f"got embedding of size {len(result[0].values)} for query")
            return result[0].values
    

        def find_matching_guides(self, query):
            """do a nearest neighbour search on firestore on the vectors"""
            query_embedding = _get_query_embedding(query)

            vector_query = self._guides.find_nearest(
                vector_field="embedding",
                query_vector=Vector(query_embedding),
                distance_measure=DistanceMeasure.COSINE,
                limit=5,
            )

            docs = list(vector_query.stream())
            logging.info(f"found {len(docs)} matching travel docs")
            for doc in docs:
                logging.info(f"title: {doc.to_dict()['title']}")
            return docs


        def find_matching_userinfo(self, query):
            """do a nearest neighbour search on firestore on the user info"""
            query_embedding = _get_query_embedding(query)

            vector_query = self._user.find_nearest(
                vector_field="embedding",
                query_vector=Vector(query_embedding),
                distance_measure=DistanceMeasure.COSINE,
                limit=5,
            )

            docs = list(vector_query.stream())
            logging.info(f"found {len(docs)} matching user docs")
            for doc in docs:
                logging.info(f"title: {doc.to_dict()['title']}")
            return docs

