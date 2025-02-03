# this is the functionality for the chat interface

import logging
import helper
from search import Search
from gemini_vertex import Gemini as GenAI
import router


class ChatBot:
    def __init__(self):
        self.search = Search()
        self.chat_model = GenAI("travel_companion", True)

    # def answer_query_basic(self, message: str, history: list): 
    #     """Generate a completion for user's query"""
    #     logging.info(f"Calling Model... session: {session_id.value}",
    #                 extra={"json_fields": {"session_id": session_id.value}})
    #     response = self.chat_model.get_response(message, history)

    #     # save chat history into firestore
    #     # helper.store_chat(session_id.value, message, response, history)
    #     return response
    

    def rewrite_travel_query(message: str, history: list, user_docs: list = []): 
        """rewrite the travel query to be more relevant for retrieval"""
        model = GenAI("travel_query_rewrite", False)
        rewrite_prompt = helper.get_input_prompt('travel_query_rewrite')
        final_prompt = rewrite_prompt.format(
            user_query=message,
            chat_history=helper.format_chat_history(history),
            user_info=helper.format_user_docs(user_docs)
        )
        return model.get_response(final_prompt)



    def rewrite_user_query(message: str, history: list): 
        """rewrite the query to be more relevant to retrieve user info"""
        model = GenAI("user_query_rewrite", False)
        rewrite_prompt = helper.get_input_prompt('user_query_rewrite')
        final_prompt = rewrite_prompt.format(
            user_query=message,
            chat_history=helper.format_chat_history(history)
        )
        return model.get_response(final_prompt)


    def respond_default(self, message: str): 
        """Use the default responder model to respond to the user"""
        model = GenAI("travel_responder", False)
        response_prompt = helper.get_input_prompt('travel_responder')
        final_prompt = response_prompt.format(
            situation="Situation: The user has either typed in something vague or open endedd, or something not related to travel. Greet the user. Respond by telling the user what you can do and how you can help.",
            user_query=message)
        return model.get_response(final_prompt)


    def respond_regret(self, message: str):
        """Respond to the user that you are unable to help them with this query"""
        model = GenAI("travel_responder", False)
        response_prompt = helper.get_input_prompt('travel_responder')
        final_prompt = response_prompt.format(
            situation="Situation: You are unable to answer the user as you don't have the relevant information needed. Thank the user for their query and ask them to try something else.",
            user_query=message)
        return self.model.get_response(final_prompt)


    def answer_query_with_rag(self, message: str, history: list): 
        """Respond to the user using the RAG approach. 
        Use the logic as defined in rag_logic.mmd"""

        # Step 1: determine if the query is related to travel
        if router.is_travel_related(message, history):
            logging.info("query is travel related. fetching travel guides")

            # Step 2: determine if the query has any user context and fetch
            if router.is_user_related(message):
                logging.info("Query contains user context, Fetching user info")
                user_query = self.rewrite_user_query(message, history)
                user_docs = self.search_model.find_matching_userinfo(user_query)
                user_info = helper.format_user_docs(user_docs)
            else:
                user_info = ""

            # Prepare final prompt
            logging.info(f"querying model for user input - {message}")
            travel_prompt = helper.get_input_prompt('travel_companion')

            # Step 3: is this a new query then fetch travel guides
            # and don't use chat history
            if router.is_new_query(message, history):
                logging.info("This seems like a fresh query, Fetching travel guides")
                travel_query = self.rewrite_travel_query(message, history, user_docs)
                travel_docs = self.search_model.find_matching_guides(travel_query)
                travel_guides = helper.format_travel_docs(travel_docs)
                final_prompt = travel_prompt.format(
                    user_query=message,
                    travel_guides=travel_guides,
                    chat_history=[],
                    user_info=user_info)

            else:
                # use chat history and no travel guides
                chat_history = helper.format_chat_history(history)
                final_prompt = travel_prompt.format(
                    user_query=message,
                    travel_guides=[],
                    chat_history=chat_history,
                    user_info=user_info)

            # Step 4: get completion from model          
            logging.info(f"Token size of prompt: {self.chat_model.get_tokens(final_prompt)}")
            llm_response = self.chat_model.get_response(prompt=final_prompt)

            # Step 5: check if the response is valid and relevant
            if router.is_response_relevant(message, llm_response):
                #Step 6: save chat history
                # helper.store_user(session_user.value, history)
                # helper.store_chat(session_id.value, message, llm_response, history)
                return llm_response
            else:
                return self.respond_regret(message)

        else:
            # query was not travel related, so prompt user
            return self.respond_default(message)
            

    # def answer_query_with_rag(message: str, history: list): 
    #     """Convert the user query to an embedding, do a match in firestore.
    #     Fetch the relevant matching docs and then pass it as context to the model to answer"""

    #     logging.info(f"Calling Model... session: {session_id.value}",
    #                 extra={"json_fields": {"session_id": session_id.value}})
    #     logging.info("user query: " + message)
    #     query = get_rewritten_query(message, history)
    #     matches = search.find_matching_docs(query)

    #     prompt = "Answer the user's travel related query below using only the provided extracts.\n\n"
    #     prompt = prompt + "User Query: " + query + "\n\n"
    #     prompt = prompt + "Travel guide extracts:\n\n"


    #     prompt = prompt + "\nAnswer:"
    #     logging.info(f"Token size of prompt: {chat_model.get_tokens(prompt)}")
    #     response = chat_model.get_response(prompt, history)

    #     # save chat history into firestore
    #     helper.store_chat(session_id.value, message, response, history)
    #     return response
