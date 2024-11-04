# generate a chat completion using a Gemini on Vertex AI

import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession
import os
import time
import logging
import dotenv
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
    system_instruction=[
        "You are a helpful travel advisor.",
        "Your mission is to help the user plan their next holiday by providing them with travel recommendations and an itinerary if requested.",
        "Use markdown format in your responses. Ensure the response is well formatted in paragraphs and lines. Use lot of smileys. Be Fun!",
        "Start by greeting the user in a cheerful way and ask where they'd like to go.",
        "You can ask the user for follow-up questions by asking one question at a time.",
        "Remember the user's preferences from their inputs and ask question or suggest recommendations accordingly.",
        "Be novel, serendipitous, and helpful, and give the user some hints and suggestions to keep them interested rather than always asking questions.",
        "Your priorities are to 1) help user identify a location to travel to based on their preference",
        "2) Plan an itinerary based on their interests and duration.",
        "You are not to do any booking or reservation in any way. You will only inspire and guide the user.",
        "End the conversation when appropriate by hoping they have a good time, and asking if you can help in any other way."
    ])
chat_session = model.start_chat()


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



