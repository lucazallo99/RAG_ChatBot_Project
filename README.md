## RAG CHATBOT ON EUROPEAN HISTORY
This is a RAG ChatBot project. It was developed as my Final for a Text Mining Class during my Master's Degree

## Overview
This project involves developing an interactive chatbot that allows users to query European history topics and receive detailed responses based on a large dataset of historical content. The chatbot uses a combination of natural language processing techniques, similarity search, and large language models (LLMs) to provide accurate and contextually relevant information. The application can search through historical documents, extract the most relevant content, and respond in real-time using AI.

## Key Features
- **Semantic Search**: Enables users to search historical documents using similarity-based search with FAISS (Facebook AI Similarity Search).
- **Embedding Models**: Utilizes Hugging Face models to generate high-dimensional vector embeddings of the document contents and queries.
- **LLM Integration**: Integrated with Google AI's generative models to produce conversational, AI-driven responses.
- **Customizable Configuration**: The chatbot's behavior, such as token limits and response temperature, can be configured through the `ChatConfig` class.
- **Contextual Conversations**: Tracks conversation history to maintain context across multiple user interactions.

## Technology Stack
- **Python**: Main programming language used for the chatbot.
- **Pandas**: For handling historical datasets.
- **FAISS**: For fast and efficient similarity searches.
- **Transformers (Hugging Face)**: To embed both queries and documents into vector space.
- **Google Generative AI**: Provides conversational AI capabilities.
- **Torch**: For handling tokenization and model inference with Hugging Face transformers.
- **Parquet**: Storage format for the historical data.

## Environment Variables
```
This project requires the following environment variables to run:

- `API_KEY`: Your OpenAI API key.
Create a `.env` file in the root directory with the following format:
API_KEY=your-key-here
```
## Project Structure
```
final-project/
├── embedding.py
├── search.py
├── llm.py
├── strategy.py
├── chatbot.py
├── ChatConfig.py
├── conversationstate.py
├── BIA6304_Zallo_Final.ipynb
└── README.md
```

## File Descriptions
- **embedding.py**: Defines the `HuggingFaceEmbedding` class to generate document embeddings and sets up FAISS for efficient similarity search. It loads historical data from a parquet file and builds a searchable FAISS index.
- **search.py**: Implements the `search()` function that accepts a query string, generates its embedding, and performs a similarity search on the historical documents using FAISS.
- **llm.py**: Contains the base class `LLMProvider` and its implementation `AIStudioProvider`, which integrates with Google's Generative AI to stream responses based on historical data and user inputs.
- **ChatConfig.py**: Manages configuration settings such as API keys, token limits, temperature, and stop sequences for controlling the chatbot's response generation.
- **strategy.py**: Implements the chatbot's interaction logic, including how user inputs are processed, when to search historical documents, and how to generate responses using the LLM.
- **chatbot.py**: The main loop for the chatbot application. Handles user interactions, passes inputs to the search and AI systems, and displays the chatbot's responses.
- **conversationstate.py**: Tracks the history of the conversation and any relevant context to ensure consistent and contextually accurate responses.

## Acknowledgements
This project was made possible through the support of several resources:
- **ChatGPT**: Used for generating code suggestions and improving development.
- **Course Material**: The concepts and code provided by the course.
- **Professor's Repository**: Inspiration and foundational code from the repository provided by my professor [bia6304-assignment5](https://github.com/54rt1n/bia6304-assignment5).
