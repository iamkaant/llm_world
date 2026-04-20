import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "google/gemma-4-31b-it:free",
    "messages": [{"role": "user", "content": "Say hi"}],
    "max_tokens": 1
}

response = requests.post(url, headers=headers, json=data)
print(f"Status Code: {response.status_code}")
print(f"Response Body: {response.text[:400]}")
