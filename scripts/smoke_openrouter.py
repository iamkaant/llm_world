import os

import requests
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in environment")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "google/gemma-4-31b-it:free",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 1,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text[:400]}")


if __name__ == "__main__":
    main()
