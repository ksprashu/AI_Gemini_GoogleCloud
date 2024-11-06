# program to generate datasets for fine-tuning
#
# Types of inputs
# 1) Simple greetings - Hi, Hello, Good morning, etc. 
# Expected outputs - Hi, hope you are doing well, how can I help you. 
# 2) I want to do X (activity - beach, trek, wildlife etc.)
# Expected output - Great, YY is the best place for <activity> <emoji>. Surely you have a location in mind? 
# 3) I want to go to X, what can I do in X (location - India, Japan, Big Sur)
# Expected output - Amazing, you can do YY <emoji> in X <emoji>. Do you need a plan? For how many days are you going?

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

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
model = GenerativeModel(
    model_name="gemini-1.5-flash-002")

def generate_greetings():
    """Call the LLM to generate about examples of greetings, and response pairs"""
    response = model.generate_content(
        contents="""
        You are a travel advisor, helping a user with their travel needs by providing ideas, inspiration, and recommendations.
        You are not a travel agent and will not help with bookings or reservations.
        Generate 100 pairs of input / output responses that will serve as training examples.
        The input is in a string format and the output is in a structured markdown format.
        The input should be a generic greeting such as Hi, Hello, Howdy etc.
        The output should be a response from a travel advisor. The response should be simple and friendly. 
        The output should be a greeting, followed by a query on how the advisor can help the user with their travel needs.
        The output should be concise and simple and not too long.
        The output should use an emoji after the greeting. The sentence after the greeting should be in a new line.
        This sentence should be a question prompting the user to answer by hinting at a travel plan.
        The output question should only be related to travel and nothing else. Absolutely nothing else.
        Generate the response as a json list. Each line of example should be a valid json dict. 
        """,
        generation_config=GenerationConfig(response_mime_type="application/json")
    )

    return json.loads(response.text)


def generate_activity_samples():
    """Call the LLM to generate example user queries for an activity, and responses"""
    response = model.generate_content(
        contents="""
        You are a travel advisor, helping a user with their travel needs by providing ideas, inspiration, and recommendations.
        You are not a travel agent and will not help with bookings or reservations.
        Generate 100 pairs of input / output responses that will serve as training examples.
        The input is in a string format and the output is in a structured markdown format.
        The input should be query from the user telling what they want to do - eg: boating, hiking, wildlife, etc.
        The input could be any form of activity that is related to a travel destination and only related to travel.
        The input could be vague and where the user is not sure what they want to do.
        The input could be a long sentence, a few words, or just a single phrase.
        The output should be a response from a travel advisor. The response should be simple and friendly. 
        The output should be an exciting acknowledgement of what the user wants to do, followed by an assertion that you have just the right thing for the user.
        The output then suggests a location, or a place, or an event related to travel that satisfies the user's needs.
        The output should be concise and simple and within 2 sentences.
        The output should use an emoji after the activity to represent what the user wants to do. 
        The recommendation should use an emoji to represent the place or location. It should be in a new line. 
        Ask the user if they like this or have a place or location in mind for this activity, or where they'd like to go for the activity.
        The sample output response in the pair should be in markdown. 
        Generate the final response of all example pairs as a json list. Each line of example should be a valid json dict. 
        """,
        generation_config=GenerationConfig(response_mime_type="application/json")
    )

    return json.loads(response.text)    


def generate_location_samples():
    """Call the LLM to generate example user queries for a location, and responses"""
    response = model.generate_content(
        contents="""
        You are a travel advisor, helping a user with their travel needs by providing ideas, inspiration, and recommendations.
        You are not a travel agent and will not help with bookings or reservations.
        Generate 100 pairs of input / output responses that will serve as training examples.
        The input is in a string format and the output is in a structured markdown format.
        The input should be query from the user telling where they want to go - a country, a city, a landmark, etc.
        The input is related to a travel destination and only related to travel.
        The input could be vague and where the user is not sure what they want to do. Eg: Maybe a continent, or maybe just "lake" or "beach".
        The input could be a long sentence, a few words, or just a single phrase.
        The output should be a response from a travel advisor. The response should be simple and friendly. 
        The output should be an exciting acknowledgement of where the user wants to go, followed by an assertion that you have just the right thing for the user.
        The output then suggests a few great things to do in this place. This should be in a new line after the acknowledgement.
        Each thing to do in the location should have an emoji to represent it, and each activity should be on a new line.
        The output should be concise and simple and within 5-7 sentences.
        Ask the user if they like this or want more options and ideas. Maybe they have an activity in mind, or something they've heard about this location.
        The sample output response in the pair should be in markdown. 
        Generate the final response of all example pairs as a json list. Each line of example should be a valid json dict. 
        """,
        generation_config=GenerationConfig(response_mime_type="application/json")
    )

    return json.loads(response.text)    
    

def write_to_file(filename, content): 
    """write the contents line by line to a file"""
    with open(filename, 'a') as f:
        for item in content:
            # each input / output pair should be written in the following format.
            contents = {
                "contents": [
                    {"role": "user",
                     "parts": [{"text": item['input']}]},
                    {"role": "model",
                     "parts": [{"text": item['output']}]}
                ]
            }
            f.write(json.dumps(contents) + '\n')


if __name__ == "__main__":
    greetings = generate_greetings()
    # print(greetings)
    write_to_file('tuning_data.jsonl', greetings)

    activity_samples = generate_activity_samples()
    # print(activity_samples)
    write_to_file('tuning_data.jsonl', activity_samples)

    location_samples = generate_location_samples()
    # print(location_samples)
    write_to_file('tuning_data.jsonl', location_samples)








