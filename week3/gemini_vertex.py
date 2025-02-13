# generate a chat completion using a Gemini on Vertex AI

import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession
import os
import time
import logging
import dotenv
import helper

# logger = logging.getLogger(__name__)

import google.cloud.logging
client = google.cloud.logging.Client()
client.setup_logging()


# fetch project_id and location from environment
dotenv.load_dotenv()
project_id = os.environ.get("PROJECT_ID")
location = os.environ.get("REGION")

# init vertex ai with project_id and location
vertexai.init(project=project_id, location=location)
model = GenerativeModel(
    model_name="gemini-1.5-flash-002",
    system_instruction=helper.get_system_prompt('gemini'))
chat_session = model.start_chat()


def get_name():
    return "Gemini 1.5 Flash"


def get_response(message: str, history: list):
    """Generate a response for user's query"""

    text_response = []
    # logging.info(f'querying gemini: {message}')
    prompt = message

    start_time = time.time() * 1000
    response = chat_session.send_message(prompt)
    endtime = time.time() * 1000

    log_data = {
        'prompt': prompt,
        'response': response.text, 
        'start_time': start_time,
        'end_time': endtime,
        'duration': endtime - start_time
    }
    logging.info(f'chat turn complete for query - {prompt}', 
                 extra={"json_fields": log_data})
    # logging.info(f'response: {response.text}')

    return response.text



