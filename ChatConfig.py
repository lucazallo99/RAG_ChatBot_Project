class ChatConfig:
    def __init__(self):
        self.api_key = None  # Will be set from environment variables
        self.max_tokens = 512
        self.temperature = 0.7
        self.stop_sequences = None
        self.chat_turn_strategy = {
            'stream': True,  # Example setting to enable streaming responses
            'log_responses': True,  # Option to log responses if needed
        }

    def load_api_key(self, api_key):
        self.api_key = api_key

    def get_config(self):
        return {
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'stop_sequences': self.stop_sequences,
            'chat_turn_strategy': self.chat_turn_strategy,
        }