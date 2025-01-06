from llm import AIStudioProvider
from search import search
from strategy import ChatTurnStrategy
from ConversationState import ConversationState  # Ensure you have this class imported

# Function to run the chatbot loop
def chatbot(api_key):
    print("Chatbot started. Type 'exit' to quit.")
    
    # Create an instance of the AIStudioProvider
    ai_provider = AIStudioProvider(api_key=api_key)

    # Create the ChatTurnStrategy instance, and pass a new instance of ChatConfig if needed
    strategy = ChatTurnStrategy(search_function=search, ai_provider=ai_provider)

    # Initialize the conversation state to track history and context
    state = ConversationState()

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Use the strategy to handle chat turns, passing history and context
        state.history = strategy.chat_turns_for(user_input, state.history, state.context)

        # Display user input and bot response from history (last two turns)
        for turn in state.history[-2:]:  # Show last two turns (user and assistant)
            print(f"{turn['role'].capitalize()}: {turn['content']}")