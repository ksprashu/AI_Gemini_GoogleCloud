# create a chat interfcace using gradio
# week1: create chat agents with gemini and gemma
# week2: include prompt management using a version controlled config file

# from gemini_tuned_vertex import TunedGemini as GenAI
from gemini_vertex import Gemini as GenAI
import vector_search
import helper

import gradio as gr

import logging
import google.cloud.logging
client = google.cloud.logging.Client()
client.setup_logging(log_level=logging.INFO)


# Use gr.State to store the session ID
session_id = gr.State(value=helper.generate_session_id())

# instantiate the model for query rewriting
rewrite_model = GenAI("gemini_tuned_rewrite", False)
# instantiate the model for chat
chat_model = GenAI("gemini_tuned", True)


def answer_query(message: str, history: list): 
    """Generate a completion for user's query"""
    logging.info(f"Calling Model... session: {session_id.value}",
                 extra={"json_fields": {"session_id": session_id.value}})
    response = chat_model.get_response(message, history)

    # save chat history into firestore
    helper.store_chat(session_id.value, message, response, history)
    return response


def get_rewritten_query(message: str):
    """use the rewrite model to rewrite the query"""
    prompt = "Rewrite the following user query and return only the rewritten text and nothing else.\nUser Query: "
    prompt = prompt + message

    rewritten_query = rewrite_model.get_response(prompt, [])
    logging.info(f"rewritten query: {rewritten_query}")

    return rewritten_query


def answer_query_with_rag(message: str, history: list): 
    """Convert the user query to an embedding, do a match in firestore.
    Fetch the relevant matching docs and then pass it as context to the model to answer"""

    logging.info(f"Calling Model... session: {session_id.value}",
                 extra={"json_fields": {"session_id": session_id.value}})
    logging.info("user query: " + message)
    query = get_rewritten_query(message)
    matches = vector_search.find_matching_docs(query)

    prompt = "Answer the user's travel related query below using only the provided extracts.\n\n"
    prompt = prompt + "User Query: " + query + "\n\n"
    prompt = prompt + "Travel guide extracts:\n\n"

    # loop at each match and append the doc title and doc text
    for doc in matches:
        prompt = prompt + doc.to_dict()['title'] + ":\n" + doc.to_dict()['text'] + "\n\n"

    prompt = prompt + "\nAnswer:"
    logging.info(f"Token size of prompt: {chat_model.get_tokens(prompt)}")
    response = chat_model.get_response(prompt, history)

    # save chat history into firestore
    helper.store_chat(session_id.value, message, response, history)
    return response


# start of app | rendering
demo = gr.ChatInterface(
    answer_query_with_rag, 
    type="messages",
    textbox=gr.Textbox(placeholder="Type in your travel query...", autofocus=True),
    title="Your Friendly Travel Advisor",
    description="Plan your travel, get inspired for your next desination, do amazing new activties. Powered by " + chat_model.get_name(),
    autofocus=True,
    examples=[
        ['Give me ideas for a beach vacation', None],
        ['I want to go on a 3 day safari in Masai Mara. When is the best time to go?', None],
        ['Build a 5-day itinerary of the best things to do in Bhutan.', None],
        ['What are the top 5 street food places in Bangalore?', None]
    ]
    )

demo.launch()
