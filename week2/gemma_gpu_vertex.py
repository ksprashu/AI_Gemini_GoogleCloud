# generate a chat completion using a Gemini on Vertex AI

from vertexai.generative_models import GenerativeModel, ChatSession
from google.cloud import aiplatform

import dotenv
import os
import logging
import helper
# logger = logging.getLogger(__name__)

import google.cloud.logging
client = google.cloud.logging.Client()
client.setup_logging()

# fetch project_id and location from environment
dotenv.load_dotenv()
project_id = os.environ.get("PROJECT_ID")
location = os.environ.get("REGION")
bucket = os.environ.get("AI_BUCKET")

# init vertex ai with project_id and location
aiplatform.init(project=project_id, location=location, staging_bucket=bucket)

# fetch the deployed endpoint
endpoint = aiplatform.Endpoint(endpoint_name="6225435935979339776")

def get_prompt(message: str, history: list) -> str:
    """Get the system instruction, history, and user prompt as a single string """
    prompt = ""
    prompt = "\n".join(helper.get_system_prompt('gemma2'))

    # insert the history as a string into the prompt
    for msg in history:
        if msg['role'] == 'user':
            prompt += "\nUser: " + msg['content']
        elif msg['role'] == 'assistant':
            prompt += "\nAssistant: " + msg['content']
    
    # finally insert the user message as the final prompt
    prompt += "\nUser: " + message
    return prompt


def get_response(message: str, history: list):
    """Generate a response for user's query"""

    logging.info(f'querying gemma2: {message}')

    # Prepare your input for prediction
    instances = [
        {"inputs": get_prompt(message, history)}
    ]

    # Make the prediction
    response = endpoint.predict(instances=instances)
    logging.info(f'response: {response.predictions}')
    print(response)
    return response.predictions

# {
#     "instances": [
#         {
#           "prompt": "What is machine learning?",
#           "max_tokens": 100
#         }
#     ]
# }