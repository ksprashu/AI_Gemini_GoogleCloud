# create a chat interfcace using gradio
import gemini_vertex as genai
import gradio as gr
import uuid
import os
import dotenv
from google.cloud import firestore

import logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO,  # Set the lowest log level to show
#                     format='%(asctime)s - %(levelname)s - %(message)s')  # Define the output format

import google.cloud.logging
client = google.cloud.logging.Client()
client.setup_logging()

# Initialize the Firestore client
dotenv.load_dotenv()
project_id = os.environ.get("PROJECT_ID")
database_id = os.environ.get("DATABASE_ID")
db = firestore.Client(project=project_id, database=database_id)


# Function to generate a unique session ID
def generate_session_id():
  """Generates a unique session ID using the UUID library."""
  return str(uuid.uuid4())


# Use gr.State to store the session ID
session_id = gr.State(value=generate_session_id())


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


def answer_query(message: str, history: list): 
    """Generate a completion for user's query"""
    logging.info(f"Calling Model... session: {session_id.value}",
                 extra={"json_fields": {"session_id": session_id.value}})
    response = genai.get_response(message, history)

    # save chat history into firestore
    store_chat(session_id.value, message, response, history)
    return response


demo = gr.ChatInterface(
    answer_query, 
    type="messages",
    textbox=gr.Textbox(placeholder="Type in your travel query..."),
    title="Your Friendly Travel Advisor",
    )


demo.launch()
