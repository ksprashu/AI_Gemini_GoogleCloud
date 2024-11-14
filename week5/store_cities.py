# This program will read the city guides information,
# convert it to embeddings and store it in firestore collection


import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from langchain_text_splitters import MarkdownHeaderTextSplitter

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector

import re
import os
import json
import dotenv
import logging
import google.cloud.logging
client = google.cloud.logging.Client()
client.setup_logging()

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

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)


def convert_to_markdown_headings(text):
    """Converts Wikivoyage headings to markdown headings."""
    for i in range(5, 0, -1):  # Iterate from 5 to 1
        pattern = f"({'=' * i})(.*?)({'=' * i})"
        # Use \2 to refer to the second capture group
        replacement = "#" * i + r" \2"
        text = re.sub(pattern, replacement, text)
    return text


def combine_metadata(metadata):
    """Combines metadata values into a comma-separated string."""
    values = list(metadata.values())
    return ", ".join(values)


def get_chunks(city_name, city_guide):
    """for each city guide, get the embeddings and save it to firestore"""

    logging.info(f"processing guide for {city_name}...")
    # first add the cityname to the start of the guide
    city_guide = f"={city_name}=\n\n" + city_guide
    # convert into markdown format
    city_guide = convert_to_markdown_headings(city_guide)
    # split the text into chunks
    texts = markdown_splitter.split_text(city_guide)

    # combine the metadata for each document and add to the text.
    for i, text in enumerate(texts):  # Use enumerate to get the index
        metadata = combine_metadata(text.metadata)
        texts[i].page_content = (
            f"{metadata}\n\n" + text.page_content
        )  # Modify the list in-place

    logging.info(f"returned {len(texts)} chunks for {city_name}")
    return texts


def get_title_for_section(metadata):
    """gets the title for the text in a chunk"""
    return f"City Guide for {metadata['Header 1']}, section {combine_metadata(metadata)}"


def get_embeddings(chunks):
    """loop through the chunks and get the embeddings"""

    logging.info(f"getting embeddings for {chunks[0].metadata['Header 1']}")

    # get the embeddings for each chunk, 10 at a time
    embeddings = []
    for i in range(0, len(chunks), 10):
        batch = chunks[i : i + 10]
        embedding_inputs = [
            TextEmbeddingInput(
                text=text.page_content,
                task_type="RETRIEVAL_DOCUMENT",
                title=get_title_for_section(text.metadata),
            )
            for text in batch
        ]
        result = model.get_embeddings(embedding_inputs)
        # collect the embeddings
        # as a dict of city, section, title, original text, embeddings
        # loop through corresponding entries of chunk and embedding
        for text, embedding in zip(batch, result):
            embeddings.append(
                {
                    "city": city_name,
                    "title": get_title_for_section(text.metadata),
                    "text": text.page_content,
                    "metadata": text.metadata,
                    "embedding": Vector(embedding.values)
                }
            )

    logging.info(f"got {len(embeddings)} embeddings for {city_name}")
    return embeddings


def save_to_firestore(embeddings):
    """save the embeddings to firestore as a doc using batch writes."""
    logging.info(f"saving {len(embeddings)} embeddings to firestore...")
    batch = db.batch()
    for embedding in embeddings:
        # Create a new document reference
        new_doc_ref = collection.document()  # Let Firestore auto-generate the ID
        batch.set(new_doc_ref, embedding)

    batch.commit()
    logging.info(f"saved {len(embeddings)} embeddings to firestore")


# main
if __name__ == "__main__":
    with open("city_guides.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            city_name = data["city"]
            city_guide = data["guide"]
            chunks = get_chunks(city_name, city_guide)
            embeddings = get_embeddings(chunks)
            save_to_firestore(embeddings)
