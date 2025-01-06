from abc import ABC, abstractmethod
from typing import List, Dict, Generator
import logging

logger = logging.getLogger(__name__)

# Base LLM Provider class using an abstract method pattern
class LLMProvider(ABC):

    @abstractmethod
    def stream_turns(self, messages: List[Dict[str, str]], config: dict) -> Generator[str, None, None]:
        pass

    @abstractmethod
    def model(self) -> str:
        pass

class AIStudioProvider(LLMProvider):
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.gem = genai.GenerativeModel(self.model)  # Access model as a property, no parentheses

    @property
    def model(self):
        # Set the model name used in the AI Studio API
        return 'gemini-1.5-flash'

    def stream_turns(self, messages: List[Dict[str, str]], config: dict) -> Generator[str, None, None]:
        from google.generativeai import GenerationConfig

        # Generation configuration
        generation_config = GenerationConfig(
            candidate_count=1,
            stop_sequences=config.get('stop_sequences', None),
            max_output_tokens=config.get('max_tokens', 512),
            temperature=config.get('temperature', 0.7)
        )

        # Google AI expects messages in a specific format: [{'role': 'user' or 'model', 'parts': ['text']}]
        formatted_messages = [
            {'role': 'user' if m['role'] == 'user' else 'model', 'parts': [m['content']]} for m in messages
        ]

        try:
            # Stream content from the generative model
            for chunk in self.gem.generate_content(formatted_messages, generation_config=generation_config, stream=True):
                # Extract the text from the candidates
                if chunk.candidates and len(chunk.candidates) > 0:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        # Join all text parts to form the full response
                        full_response = ''.join(part.text for part in candidate.content.parts)
                        yield full_response.strip()
                    else:
                        logger.error(f"Content parts not found in candidate: {candidate}")
                        yield "No valid response could be generated."
                else:
                    logger.error("No candidates found in the response.")
                    yield "No response could be generated."
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            yield "An unexpected error occurred."