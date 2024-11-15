# generate a chat completion using a Gemini on Vertex AI

import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession
import os
import time
import logging
import dotenv
import helper

# logger = logging.getLogger(__name__)

class TunedGemini():
    """Class that implements methods to invoke the Gemini model"""

    _chat_session = None
    _model = None

    def __init__(self, prompt_name: str, is_chat: bool = True):
        """initialise vertex and other variables"""

        # fetch project_id and location from environment
        dotenv.load_dotenv()
        project_id = os.environ.get("PROJECT_ID")
        location = os.environ.get("REGION")

        # init vertex ai with project_id and location
        vertexai.init(project=project_id, location=location)
        self._model = GenerativeModel(
            model_name=self._get_model_endpont('5404654903891066880', project_id, location),
            system_instruction=helper.get_system_prompt(prompt_name))

        if is_chat:
            self._chat_session = self._model.start_chat()


    def _get_model_endpont(self, endpoint_id: int, project_id: str, location: str):
        """build then endpoint in the following format.
        projects/<projectid/locations/<region>/endpoints/<endpointid>
        """
        logging.info(f"endpoint_id: {endpoint_id}")
        return f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"


    def get_name(self):
        return "Gemini Travel-tuned model"


    def get_tokens(self, message: str):
        """return the input tokens size"""
        return self._model.count_tokens(message).total_tokens


    def get_response(self, message: str, history: list):
        """Generate a response for user's query"""

        # logging.info(f'querying gemini: {message}')
        prompt = message

        start_time = time.time() * 1000
        if self._chat_session is None:
            response = self._model.generate_content(prompt)
        else:   
            response = self._chat_session.send_message(prompt)
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
        logging.debug('chat history: ', extra={"json_fields": history})

        return response.text


    def start_new_chat(self):
        """erases the current session and starts a new chat session"""
        self._chat_session = self._model.start_chat()
