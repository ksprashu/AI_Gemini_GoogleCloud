# This is a helper class the provides reusable functions.
import uuid
import dotenv
import os
import json
from google.cloud import firestore

import logging
import google.cloud.logging
client = google.cloud.logging.Client()
client.setup_logging()


# Initialize the Firestore client
dotenv.load_dotenv()
project_id = os.environ.get("PROJECT_ID")
database_id = os.environ.get("DATABASE_ID")
db = firestore.Client(project=project_id, database=database_id)


# Function to store the chat to Firestore
def store_chat(session_id, message, response, history):
    """Stores the chat history in Firestore and updates it against a UUID."""

    try:
        doc_ref = db.collection("chat_history").document(session_id)
        # append message and response to a copy of history
        chat_messages = history.copy()
        chat_messages.append({"role": "user", "content": message})
        chat_messages.append({"role": "assistant", "content": response})

        # save chat_messages to firestore and update existing data in document
        doc_ref.set({"chat_messages": chat_messages}, merge=True)
        logging.info(f"Chat history stored successfully for session: {session_id}")
    except Exception as e:
        logging.error(f"Error storing chat history: {e}")


# Function to generate a unique session ID
def generate_session_id():
  """Generates a unique session ID using the UUID library."""
  return str(uuid.uuid4())


# Gets the latest system prompt for a given model
def get_system_prompt(model_name):
    """Gets the latest system prompt for a given model."""
    with open('prompts.json') as f:
        data = json.load(f)
    
    # find the latest version of the prompt
    latest_prompt = None
    for prompt_data in data[model_name]:
        if latest_prompt is None or prompt_data['version'] > latest_prompt['version']:
            latest_prompt = prompt_data
    
    return "\n".join(latest_prompt['system_prompt'])

