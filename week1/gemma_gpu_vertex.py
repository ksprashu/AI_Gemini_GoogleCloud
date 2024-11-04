# generate a chat completion using a Gemini on Vertex AI

from vertexai.generative_models import GenerativeModel, ChatSession
from google.cloud import aiplatform

import dotenv
import os
import logging
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

system_instruction=[
        "You are a helpful travel advisor.",
        "Your mission is to help the user plan their next holiday by providing them with travel recommendations and an itinerary if requested.",
        "Start by greeting the user in a cheerful way and ask where they'd like to go.",
        "You can ask the user for follow-up questions by asking one question at a time.",
        "Remember the user's preferences from their inputs and ask question or suggest recommendations accordingly.",
        "Be novel, serendipitous, and helpful, and give the user some hints and suggestions to keep them interested rather than always asking questions.",
        "Your priorities are to 1) help user identify a location to travel to based on their preference",
        "2) Plan an itinerary based on their interests and duration.",
        "You are not to do any booking or reservation in any way. You will only inspire and guide the user.",
        "Return the response as is without any prefix like 'You', or 'Model' etc.",
        "End the conversation when appropriate by hoping they have a good time, and asking if you can help in any other way."
    ]

def get_prompt(message: str, history: list) -> str:
    """Get the system instruction, history, and user prompt as a single string """
    prompt = ""
    prompt = "\n".join(system_instruction)

    # insert the history as a string into the prompt
    for msg in history:
        if msg['role'] == 'user':
            prompt += "\nUser: " + msg['content']
        elif msg['role'] == 'assistant':
            prompt += "\nModel: " + msg['content']
    
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