# models/llm_model.py

from config import GROQ_API_KEY
from groq import Groq  # Ensure that the Groq LLM client library is installed

class LLMModel:
    def __init__(self):
        # Initialize the Groq client with your API key.
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate(self, prompt: str) -> str:
        """
        Generate a response using the Groq LLM.
        """
        try:
            completion = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are an expert in government RFP analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=True,
                stop=None,
            )
            
            # Iterate over the streaming response and concatenate text segments.
            complete_response = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                print(content, end="")  # Optionally print the streaming response in real-time.
                complete_response += content
            
            return complete_response
        except Exception as e:
            return f"Error generating response: {e}"
