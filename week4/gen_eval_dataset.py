# generate good and bad datasets for eval


import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import gemini_vertex as model_gemini
import gemini_tuned_vertex as model_tuned
import gemma_gpu_vertex as model_gemma


import os
import json
import dotenv
import time

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
    model_name="gemini-1.5-pro-002")


def generate_eval_datasets():
    """Generate sample chat responses for greetings and questions"""
    response = model.generate_content(
        contents="""
        Randomly behave as a friendly tourist, a businessman, a bored individual, and obnoxious politician, etc. 
        These are just example personas, create 20 different personas and ask your travel questions or recommendations to the advisor as a persona picked at random for each sample query.
        You may be from anywhere in the world. You may want to do anything in the world. 
        You may also just be trying to have a conversation or irritate the advisor, without having any specific travel requirements. 
        You may have a very long winding sentence as a query, or just one or two words.
        Using all these personas, generate a sample set of 60 travel related queries that I can ask to the travel advisor. 
        Generate the response as a json list of strings. 
        """,
        generation_config=GenerationConfig(response_mime_type="application/json")
    )

    return json.loads(response.text)    


def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█', print_end='\r'):
    """
    Prints a text-based progress bar.

    Args:
        iteration (int): Current iteration (out of 'total').
        total (int): Total number of iterations.
        prefix (str, optional): Prefix string. Defaults to ''.
        suffix (str, optional): Suffix string. Defaults to ''.
        length (int, optional): Character length of bar. Defaults to 50.
        fill (str, optional): Bar fill character. Defaults to '█'.
        print_end (str, optional): End character (e.g. '\r', '\r\n'). Defaults to '\r'.
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print a newline on completion
    if iteration == total: 
        print()



def get_eval_responses(model, eval_queries):
    """read the dataset of eval queries and run it against the model"""
    eval_responses = []
    # display an ascii progress bar for the queries
    print(f"Evaluating queries for model: {model.get_name()}")
    total_items = len(eval_queries)
    for i, query in enumerate(eval_queries):
        try:
            response = model.get_response(query, [])
            eval_responses.append({"query": query, "response": response})
        except Exception as e:
            print(f"Error generating response for query {query}: {e}")
            eval_responses.append({"query": query, "response": "Error generating response"})

        print_progress_bar(i + 1, total_items, prefix='Progress:', suffix='Complete', length=50)
        time.sleep(1) # sleep for a second to avoid quota errors

    return eval_responses


if __name__ == "__main__":
    # eval_queries = generate_eval_datasets()
    # # write to a file
    # with open('eval_queries.json', 'w') as f:
    #     json.dump(eval_queries, f)

    # read the queries from the files
    with open('eval_queries.json', 'r') as f:
        eval_queries = json.load(f)

    # run the query against each model
    # gemini_responses = get_eval_responses(model_gemini, eval_queries)
    # with open('gemini_responses.json', 'w') as f:
    #     json.dump(gemini_responses, f)

    # tuned_responses = get_eval_responses(model_tuned, eval_queries)
    # with open('tuned_responses.json', 'w') as f:
    #     json.dump(tuned_responses, f)

    gemma_responses = get_eval_responses(model_gemma, eval_queries)
    with open('gemma_responses.json', 'w') as f:
        json.dump(gemma_responses, f)



