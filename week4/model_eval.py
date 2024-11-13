# here we'll evaluate our gemma model and gemini tuned model against gemini

import pandas as pd

import vertexai
from vertexai.evaluation import (
    EvalTask,
    PairwiseMetric,
    PairwiseMetricPromptTemplate,
    PointwiseMetric,
    PointwiseMetricPromptTemplate,
    MetricPromptTemplateExamples
)

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


def get_responses(filename):
    """load the response file and return the prompt, response from the list as a dataframe"""
    with open(filename, 'r') as f:
        responses = json.load(f)

    # responses = [response['response'] for response in responses]
    # eval_dataset = pd.DataFrame({
    #     "response" : responses,
    # })

    eval_dataset = pd.DataFrame({
        "prompt": [response['prompt'] for response in responses],
        "response": [response['response'] for response in responses],
    })
    
    return eval_dataset


def create_eval_task(df, model_name):
    """Create an evaluation task for the given dataframe"""
    eval_task = EvalTask(
        dataset=df,
        experiment=f"{model_name}-eval",
        metrics=[custom_text_quality]
        )

    return eval_task


def run_pointwise_eval():
    """Run an evaluation for the responses"""
    gemini_df = get_responses('gemini_responses.json')
    gemini_eval = create_eval_task(gemini_df, 'gemini-flash')
    print(f"Gemini evaluation task created: {gemini_eval.experiment}")
    gemini_results = gemini_eval.evaluate()
    print("Summary Metrics:")
    print(gemini_results.summary_metrics)
    print("Metrics Table:")
    print(gemini_results.metrics_table)
    

    tuned_df = get_responses('tuned_responses.json')
    tuned_eval = create_eval_task(tuned_df, 'tuned-gemini')
    print(f"Tuned evaluation task created: {tuned_eval.experiment}")
    tuned_results = tuned_eval.evaluate()
    print("Summary Metrics:")
    print(tuned_results.summary_metrics)
    print("Metrics Table:")
    print(tuned_results.metrics_table)


    gemma_df = get_responses('gemma_responses.json') 
    gemma_eval = create_eval_task(gemma_df, 'gemma2-9b-it')
    print(f"Gemma evaluation task created: {gemma_eval.experiment}")
    gemma_results = gemma_eval.evaluate()
    print("Summary Metrics:")
    print(gemma_results.summary_metrics)
    print("Metrics Table:")
    print(gemma_results.metrics_table)


if __name__ == "__main__":
    run_pointwise_eval()

