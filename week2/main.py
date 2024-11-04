# create a chat interfcace using gradio
# week2: include prompt management using a version controlled config file

import gemini_vertex as genai
import gradio as gr

import helper

import logging
import google.cloud.logging
client = google.cloud.logging.Client()
client.setup_logging()


# Use gr.State to store the session ID
session_id = gr.State(value=helper.generate_session_id())


def answer_query(message: str, history: list): 
    """Generate a completion for user's query"""
    logging.info(f"Calling Model... session: {session_id.value}",
                 extra={"json_fields": {"session_id": session_id.value}})
    response = genai.get_response(message, history)

    # save chat history into firestore
    helper.store_chat(session_id.value, message, response, history)
    return response


demo = gr.ChatInterface(
    answer_query, 
    type="messages",
    textbox=gr.Textbox(placeholder="Type in your travel query..."),
    title="Your Friendly Travel Advisor",
    )


demo.launch()
