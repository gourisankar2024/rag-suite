import requests
import os
import logging

class GroqClient:
    def __init__(self, plm):
        os.environ["GROQ_API_KEY"] = ''
        # Fetch API Key from environment variables for security
        api_key = os.getenv("GROQ_API_KEY")  # Fetch from environment
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Please add it in Hugging Face Secrets.")
        
          # Explicitly set it
        self.api_key = api_key  
        self.model = plm
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
    def generate(self, text, temperature=0.7, system=""):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": text})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }

        try:
            # Adding a timeout to avoid hanging indefinitely
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx, 5xx)

            response_json = response.json()
            if "choices" in response_json and response_json["choices"]:
                return response_json["choices"][0].get("message", {}).get("content", "")
            else:
                self.logger.error(f"Unexpected response format: {response_json}")
                return ""

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return ""

