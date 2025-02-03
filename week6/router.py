# this file contains a bunch of routers which use LLMs to make decisions

from gemini_vertex import Gemini as GenAI
import helper
import logging


def is_travel_related(message: str, history: list) -> bool:
    """Checks with the LLM if the prompt is related to travel"""
    use_case = "travel_query_check"
    model = GenAI("travel_query_check", is_chat=False)
    input_prompt = helper.get_input_prompt(use_case)
    final_prompt = input_prompt.format(
        user_query=message,
        chat_history=helper.format_chat_history(history))
    response = model.get_response(final_prompt)
    return True if "true" in response.lower() else False


def is_user_related(message: str) -> bool:
    """Checks with the LLM if the prompt is related to the user"""
    use_case = "user_query_check"
    model = GenAI("user_query_check", is_chat=False)
    input_prompt = helper.get_input_prompt(use_case)
    final_prompt = input_prompt.format(user_query=message)
    response = model.get_response(final_prompt)
    return True if "true" in response.lower() else False


def is_response_relevant(message: str, llm_response: str) -> bool:
    """Checks with the LLM if the response is related to travel"""
    use_case = "travel_response_check"
    model = GenAI("travel_response_check", is_chat=False)
    input_prompt = helper.get_input_prompt(use_case)
    final_prompt = input_prompt.format(
        user_query=message, llm_response=llm_response)
    response = model.get_response(final_prompt)
    return True if "true" in response.lower() else False


def is_new_query(message: str, history: list) -> bool:
    """Checks with the LLM if the prompt is related to travel"""
    use_case = "new_query_check"
    model = GenAI("new_query_check", is_chat=False)
    input_prompt = helper.get_input_prompt(use_case)
    final_prompt = input_prompt.format(
        user_query=message, 
        chat_history=helper.format_chat_history(history))
    response = model.get_response(final_prompt)
    return True if "true" in response.lower() else False

