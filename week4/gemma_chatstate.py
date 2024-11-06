# ChatState class for a conversation with Gemma

class ChatState():
  """
  Manages the conversation history for a turn-based chatbot
  Follows the turn-based conversation guidelines for the Gemma family of models
  documented at https://ai.google.dev/gemma/docs/formatting
  """

  __START_TURN_USER__ = "<start_of_turn>user\n"
  __START_TURN_MODEL__ = "<start_of_turn>model\n"
  __END_TURN__ = "<end_of_turn>\n"

  def __init__(self, model, system=""):
    """
    Initializes the chat state.

    Args:
        model: The language model to use for generating responses.
        system: (Optional) System instructions or bot description.
    """
    self.model = model
    self.system = system
    self.history = []

  def add_to_history_as_user(self, message):
      """
      Adds a user message to the history with start/end turn markers.
      """
      self.history.append(self.__START_TURN_USER__ + message + self.__END_TURN__)

  def add_to_history_as_model(self, message):
      """
      Adds a model response to the history with start/end turn markers.
      """
      self.history.append(self.__START_TURN_MODEL__ + message + self.__END_TURN__)

  def get_history(self):
      """
      Returns the entire chat history as a single string.
      """
      return "".join([*self.history])

  def get_full_prompt(self):
    """
    Builds the prompt for the language model, including history and system description.
    """
    prompt = self.get_history() + self.__START_TURN_MODEL__
    if len(self.system)>0:
      prompt = self.system + "\n" + prompt
    return prompt

  def send_message(self, message):
    """
    Handles sending a user message and getting a model response.

    Args:
        message: The user's message.

    Returns:
        The model's response.
    """
    self.add_to_history_as_user(message)
    prompt = self.get_full_prompt()

    # Prepare your input for prediction
    instances = [
        {
            "inputs": prompt,
            "max_tokens": 8192,
            "max_decode_steps": 8192,
            "parameters": {
                "max_tokens": 8192,
                "max_decode_steps": 8192,
                "temperature": 0.7
            }
        }
    ]

    response = self.model.predict(instances=instances)
    print(response)

    result = response.predictions[0].strip()
    self.add_to_history_as_model(result)
    return result
  
