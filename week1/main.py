# create a chat interfcace using gradio
import gemini_vertex as genai
import gradio as gr

import logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO,  # Set the lowest log level to show
#                     format='%(asctime)s - %(levelname)s - %(message)s')  # Define the output format

import google.cloud.logging
client = google.cloud.logging.Client()
client.setup_logging()

def answer_query(message: str, history: list): 
    """Generate a completion for user's query"""
    # logging.info(f'user: {message}')
    # logger.info(f'history: {history}')
    logging.info("Calling Model...")
    response = genai.get_response(message, history)
    return response
    

demo = gr.ChatInterface(
    answer_query, 
    type="messages",
    textbox=gr.Textbox(placeholder="Type in your travel query..."),
    title="Your Friendly Travel Advisor",
    )

demo.launch()
