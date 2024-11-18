# This creates a custom DeepEval evaluator using Gemini 1.5 Pro

from deepeval.models import DeepEvalBaseLLM

import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession
import os
import logging
import dotenv

class Custom_GeminiPro(DeepEvalBaseLLM):
    def __init__(self):
        # fetch project_id and location from environment
        dotenv.load_dotenv()
        project_id = os.environ.get("PROJECT_ID")
        location = os.environ.get("REGION")

        # init vertex ai with project_id and location
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name="gemini-1.5-pro-002")

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        response = model.generate_content(prompt)
        return response.text

    async def a_generate(self, prompt: str) -> any:
        # model = self.load_model()
        # responses = model.generate_content(prompt, stream=True)
        # for chunk in responses:
        #     if chunk.text == "[DONE]":
        #         break
        #     yield chunk.text
        return self.generate(prompt)

    def get_model_name(self):
        return "Gemini 1.5 Pro"
    
