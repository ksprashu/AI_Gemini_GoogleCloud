# here we'll evaluate our gemma model and gemini tuned model against gemini

import pandas as pd

import vertexai
from vertexai.evaluation import EvalTask, PointwiseMetric, PointwiseMetricPromptTemplate


import os
import json
import dotenv

# import google.cloud.logging
# client = google.cloud.logging.Client()
# client.setup_logging()

# fetch project_id and location from environment
dotenv.load_dotenv()
project_id = os.environ.get("PROJECT_ID")
location = os.environ.get("REGION")

# init vertex ai with project_id and location
vertexai.init(project=project_id, location=location)

custom_text_quality = PointwiseMetric(
    metric="custom_text_quality",
    metric_prompt_template=PointwiseMetricPromptTemplate(
        criteria={
            "fluency": (
                "Sentences flow smoothly and are easy to read, avoiding awkward"
                " phrasing or run-on sentences. Ideas and sentences connect"
                " logically, using transitions effectively where needed."
            ),
            "entertaining": (
                "Short, amusing text that incorporates emojis, exclamations and"
                " questions to convey quick and spontaneous communication and"
                " diversion."
            ),
        },
        rating_rubric={
            "1": "The response performs well on both criteria.",
            "0": "The response is somewhat aligned with both criteria",
            "-1": "The response falls short on both criteria",
        },
    ),
)

def run_eval():
    """Run an evaluation for the responses"""
    