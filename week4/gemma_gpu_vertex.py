# generate a chat completion using a Gemini on Vertex AI

from vertexai.generative_models import GenerativeModel, ChatSession
from google.cloud import aiplatform

from gemma_chatstate import ChatState

import time
import dotenv
import os
import logging
import helper

# logger = logging.getLogger(__name__)

# fetch project_id and location from environment
dotenv.load_dotenv()
project_id = os.environ.get("PROJECT_ID")
location = os.environ.get("REGION")
bucket = os.environ.get("AI_BUCKET")

# init vertex ai with project_id and location
aiplatform.init(project=project_id, location=location, staging_bucket=bucket)

# fetch the deployed endpoint
endpoint = aiplatform.Endpoint(endpoint_name="6225435935979339776")
chat_model = ChatState(model=endpoint, system=helper.get_system_prompt('gemma2'))


def get_name():
    return "Gemma2 9b-it"


def get_response(message: str, history: list):
    """Generate a response for user's query"""

    # Make the prediction
    start_time = time.time() * 1000
    response = chat_model.send_message(message)
    endtime = time.time() * 1000 

    log_data = {
        'prompt': message,
        'response': response, 
        'start_time': start_time,
        'end_time': endtime,
        'duration': endtime - start_time
    }

    logging.info(f'chat turn complete for query - {message}', 
                 extra={"json_fields": log_data})
    logging.debug('chat history: ', extra={"json_fields": history})

    return response


def start_new_chat():
    """erases the current session and starts a new chat session"""
    global chat_model
    chat_model = ChatState(model=endpoint, system=helper.get_system_prompt('gemma2'))


# {
#     "instances": [
#         {
#           "prompt": "What is machine learning?",
#           "max_tokens": 100
#         }
#     ]
# }
