# Tune a Gemini Flash model according to the tuning data generated.
# https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-use-supervised-tuning

import time

import vertexai
from vertexai.tuning import sft

import os
import dotenv
import json

# fetch project_id and location from environment
dotenv.load_dotenv()
project_id = os.environ.get("PROJECT_ID")
location = os.environ.get("REGION")

# init vertex ai with project_id and location
vertexai.init(project=project_id, location=location)

sft_tuning_job = sft.train(
    source_model="gemini-1.5-flash-002",
    train_dataset="gs://ksp-l100-ai/tuning_data.jsonl",
    # The following parameters are optional
    # validation_dataset="gs://cloud-samples-data/ai-platform/generative_ai/gemini-1_5/text/sft_validation_data.jsonl",
    epochs=300,
    adapter_size=4,
    learning_rate_multiplier=1.0,
    tuned_model_display_name="travel_tuned_gemini_1_5_flash",
)

# Polling for job completion
while not sft_tuning_job.has_ended:
    time.sleep(60)
    sft_tuning_job.refresh()

print(sft_tuning_job.tuned_model_name)
print(sft_tuning_job.tuned_model_endpoint_name)
print(sft_tuning_job.experiment)

