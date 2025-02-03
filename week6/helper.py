# This is a helper class the provides reusable functions.
import uuid
import dotenv
import os
import json
from google.cloud import firestore

import logging

# Initialize the Firestore client
dotenv.load_dotenv()
project_id = os.environ.get("PROJECT_ID")
database_id = os.environ.get("DATABASE_ID")
db = firestore.Client(project=project_id, database=database_id)


# Function to store the chat to Firestore
def store_chat(session_id, message, response, history):
    """Stores the chat history in Firestore and updates it against a UUID."""

    try:
        doc_ref = db.collection("chat-history").document(session_id)
        # append message and response to a copy of history
        chat_messages = history.copy()
        chat_messages.append({"role": "user", "content": message})
        chat_messages.append({"role": "assistant", "content": response})

        # save chat_messages to firestore and update existing data in document
        doc_ref.set({"chat_messages": chat_messages}, merge=True)
        logging.info(f"Chat history stored successfully for session: {session_id}")
    except Exception as e:
        logging.error(f"Error storing chat history: {e}")


# Function to fetch the entire chat history from Firestore
def fetch_chats():
    """Fetch the entire chat history from Firestore."""
    try:
        docs = db.collection("chat-history").stream()
        chat_history = []
        for doc in docs:
            # collect only the docs where the role is "user"
            chat_messages = doc.to_dict()['chat_messages']
            chat_history = [msg['content'] for msg in chat_messages if msg['role'] == 'user']

        logging.info(f"fetched {len(chat_history)} chat history records")
        return chat_history
    
    except Exception as e:
        logging.error(f"Error fetching chat history: {e}")
        return []


# Function to generate a unique session ID
def generate_session_id():
  """Generates a unique session ID using the UUID library."""
  return str(uuid.uuid4())


# Function to get latest prompts for a given use case
def _get_use_case(use_case):
    """Gets the latest prompts for a given use case."""
    try:
        # Get the directory of helper.py
        helper_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to config.json
        config_path = os.path.join(helper_dir, 'prompts.json')

        with open(config_path) as f:
            data = json.load(f)
        
        # find the latest version of the prompt
        latest_prompt = None
        for prompt_data in data[use_case]:
            if latest_prompt is None or prompt_data['version'] > latest_prompt['version']:
                latest_prompt = prompt_data
        
        return latest_prompt
    except Exception as e:
        logging.error(f"Error getting use case: {e}")
        return None


# Gets the latest system prompt for a given model
def get_system_prompt(use_case):
    """Gets the latest system prompt for a given model."""
    prompt = _get_use_case(use_case)
    if prompt is not None:
        return "\n".join(prompt['system_prompt'])
    else:
        return None


# Gets the latest query prompt for the given use case
def get_input_prompt(use_case):
    """Gets the latest query prompt for the given use case."""
    prompt = _get_use_case(use_case)
    if prompt is not None:
        return "\n".join(prompt['input_prompt'])
    else:
        return None
    
def format_chat_history(history: list) -> str: 
    """merge the chat history into a single string
    The content from the user and model (assistant) is merged together
    in the following format
    User: <user content>
    Model: <model content>
    """
    formatted_history = ""
    for turn in history:
        if turn['role'] == 'user':
            formatted_history += f"User: {turn['content']}\n"
        else:
            formatted_history += f"Model: {turn['content']}\n"

    return formatted_history


def format_user_docs(user_docs: list) -> str:
    """merge the user docs into a single string
    """
    pass


def format_travel_docs(travel_docs: list) -> str:
    """merge the travel docs into a single string
    """
    # loop at each match and append the doc title and doc text
    for doc in travel_docs:
        travel_guide = travel_guide + doc.to_dict()['title'] + ":\n" + doc.to_dict()['text'] + "\n\n"

    return travel_guide
