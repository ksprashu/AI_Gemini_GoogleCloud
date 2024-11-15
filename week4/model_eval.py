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
            "factual_accuracy": (
                "The response provides accurate and verifiable information."
            ),
            "conciseness": (
                "The response avoids unnecessary information and gets to the point."
            ),
            "fluency": ( 
                "Sentences are clear, well-structured, and easy to read."
            ),
            "engagement": (
                "The response is engaging and keeps the user interested." 
            ),
            "relevance": (
                "The response directly addresses the user's question."
            )
        },
        rating_rubric={
            "5": "Excellent - Meets all criteria exceptionally well.",
            "4": "Good - Meets most criteria with minor issues.",
            "3": "Fair - Meets some criteria, but with room for improvement.",
            "2": "Poor - Meets few criteria, significant issues.",
            "1": "Very Poor - Barely meets any criteria." 
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

