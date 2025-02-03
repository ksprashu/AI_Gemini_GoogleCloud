# create a chat interfcace using gradio
# week1: create chat agents with gemini and gemma
# week2: include prompt management using a version controlled config file

# from gemini_tuned_vertex import TunedGemini as GenAI
from gemini_vertex import Gemini as GenAI
from chatbot import ChatBot
import helper

import gradio as gr

import logging
import google.cloud.logging

# Set the root logger level *first*
logging.getLogger().setLevel(logging.INFO)

# setup the cloud logging handler
client = google.cloud.logging.Client()
client.setup_logging(log_level=logging.INFO)

# Add a StreamHandler for console output
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)

# Print the handlers
logger = logging.getLogger()
for handler in logger.handlers:
    print(handler)

# State variables
session_id = gr.State(value=helper.generate_session_id())
user_name = gr.State(value="")  # Initially empty

chat = ChatBot()
# def answer_query_with_rag(message: str, history: list):
#     """call the chat model to get a response"""
#     return chat.answer_query_with_rag(message, history)


# start of app | rendering
# demo = gr.ChatInterface(
#     chat.answer_query_with_rag,
#     type="messages",
#     textbox=gr.Textbox(placeholder="Type in your travel query...", autofocus=True),
#     title="Your Friendly Travel Advisor",
#     description="Plan your travel, get inspired for your next desination, do amazing new activties",
#     autofocus=True,
#     examples=[
#         ["Give me ideas for a beach vacation", None],
#         ["I want to go on a 3 day safari in Masai Mara. When is the best time to go?", None],
#         ["Build a 2-day itinerary of the best things to do in Bhutan.", None],
#         ["What are the top 5 street food places in Mumbai?", None],
#     ],
# )
# demo.launch()


def set_user_name(name):
    # Log name to firestore or other persistent storage if needed
    logging.info(f"Setting user name to: {name}")
    return name

css = """
h1 {
    text-align: center;
    display:block;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Your Friendly Travel Advisor")
    gr.HTML("<center>Plan your travel, get inspired for your next desination, do amazing new activties.</center>")

    user_name_input = gr.Textbox(label="Your Name/Handle (optional: to preserve state)", lines=1)
    user_name_input.change(set_user_name, inputs=user_name_input, outputs=user_name) # Update state on change

#     chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages")

#     chat_input = gr.MultimodalTextbox(
#         interactive=True,
#         file_count="multiple",
#         placeholder="Type in your travel query...",
#         show_label=False,
#         autofocus=True,
#     )

# chat_input = gr.Textbox(
#     interactive=True,
#     placeholder="Type in your travel query...",
#     show_label=False,
#     autofocus=True,
#     # container=False,
# )


# msg = gr.Textbox(placeholder="Type in your travel query...", show_label=False)
# clear = gr.ClearButton([msg, chatbot])


# chat_input.submit(
#     chat.answer_query_with_rag,
#     inputs=[chat_input, chatbot],
#     outputs=chatbot)

# clear = gr.Button("New Chat")
# clear.click(new_chat, outputs=[session_id, user_name, chatbot], queue=False)  # Clear outputs and reset session

demo.launch()
