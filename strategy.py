import logging
from ChatConfig import ChatConfig  # Ensure to import ChatConfig
from llm import AIStudioProvider

# Initialize the logger for this module
logger = logging.getLogger(__name__)

class ChatTurnStrategy:
    def __init__(self, search_function, chat_config: ChatConfig):
        self.search = search_function
        self.chat_config = chat_config  # Store chat configuration
        self.ai_provider = AIStudioProvider(api_key=self.chat_config.api_key)  # Initialize AI provider here

    def user_turn_for(self, user_input: str) -> dict:
        return {"role": "user", "content": user_input}

    def handle_greetings(self, user_input: str) -> bool:
        """Check if the input is a greeting and return True if it is."""
        greetings = ["hello", "hi", "hey", "greetings", "hola", "ciao"]
        for greeting in greetings:
            if user_input.lower().startswith(greeting):
                return True
        return False

    def chat_turns_for(self, user_input: str, history: list = [], context: str = "") -> list:
        user_turn = self.user_turn_for(user_input)

        # Handle greetings first
        if self.handle_greetings(user_input):
            assistant_turn = {
                "role": "assistant",
                "content": "Hello! How can I assist you today?"
            }
            return history + [user_turn, assistant_turn]

        # Perform the search query for non-greeting inputs
        search_results = self.search(user_input, top_k=3)

        if not search_results.empty:
            # Compile the content from search results
            doc_content = " ".join(search_results['content'].values)

            # Generate a refined response
            response_text = self.get_ai_response(user_input, doc_content)

            assistant_turn = {
                "role": "assistant",
                "content": response_text.strip() or "I couldn't generate a response."
            }
        else:
            assistant_turn = {
                "role": "assistant",
                "content": "I couldn't find relevant information in the documents."
            }

        return history + [user_turn, assistant_turn]

    def get_ai_response(self, user_input: str, doc_content: str, history: list = []) -> str:
        try:
            # Combine user input and document content into a prompt, including conversation history
            history_text = "\n".join([f"{entry['role']}: {entry['content']}" for entry in history])
            prompt = f"{history_text}\nUser question: {user_input}\n\nRelevant historical data: {doc_content}\n\nGenerate a response based on the information above."
            
            # Create a message list (in Google AI Studio's format)
            messages = [{"role": "user", "content": prompt}]
            
            # Get the configuration for generation
            config = self.chat_config.get_config()  # Use the ChatConfig instance to retrieve settings
            
            # Stream the response from the provider
            response_text = ""
            for chunk in self.ai_provider.stream_turns(messages, config):
                response_text += " " + chunk.strip()
            
            return response_text.strip()
        except Exception as e:
            logger.error(f"Error during AI response generation: {str(e)}")
            return "An error occurred while generating a response."
