# generate a chat completion using a Gemini on Vertex AI

import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession
import os
import time
import logging
import dotenv
import helper

# logger = logging.getLogger(__name__)

class Gemini():
    """Class that implements methods to invoke the Gemini model"""

    def __init__(self, use_case: str, is_chat: bool = True, model_endpoint: str = None):
        """initialise vertex and other variables"""

        logging.info(f"creating a {'DEFAULT' if model_endpoint is not None else 'TUNED'} "+
                     f"{'Chat' if is_chat else 'Query'} model for {use_case}")
        # fetch project_id and location from environment
        dotenv.load_dotenv()
        project_id = os.environ.get("PROJECT_ID")
        location = os.environ.get("REGION")

        # init vertex ai with project_id and location
        vertexai.init(project=project_id, location=location)
        if model_endpoint is None:
            self._model = GenerativeModel(
                model_name="gemini-1.5-flash-002",
                system_instruction=helper.get_system_prompt(use_case))
        else:
            GenerativeModel(
                model_name=self._get_model_endpont('5404654903891066880', project_id, location),
                system_instruction=helper.get_system_prompt(use_case))

        if is_chat:
            self._chat_session = self._model.start_chat()
        else:
            self._chat_session = None


    def get_name(self):
        return "Gemini 1.5 Flash"


    def get_tokens(self, message: str):
        """return the input tokens size"""
        return self._model.count_tokens(message).total_tokens


    def get_response(self, prompt: str):
        """Generate a response for user's query"""

        # logging.info(f'querying gemini: {message}')
        start_time = time.time() * 1000
        if self._chat_session is None:
            logging.info("Generating content without chat model")
            response = self._model.generate_content(prompt)
        else:   
            logging.info("Generating content with chat model")
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

        return response.text


    def start_new_chat(self):
        """erases the current session and starts a new chat session"""
        self._chat_session = self._model.start_chat()

