class ConversationState:
    def __init__(self):
        self.history = []  # Stores user inputs and bot responses
        self.context = ""  # Keeps relevant context from previous turns
        self.need_clarification = False  # Tracks if clarification is needed

    def update_history(self, user_input, bot_response):
        self.history.append({
            "user_input": user_input,
            "bot_response": bot_response
        })

    def add_context(self, context_data):
        self.context += "\n" + context_data  # Append context as conversation progresses

    def clear_context(self):
        self.context = ""  # Clear context if the topic changes significantly

