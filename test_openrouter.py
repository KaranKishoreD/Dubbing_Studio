
import os
import requests

key = os.environ.get("OPENROUTER_API_KEY")
if not key:
    raise RuntimeError("API key not set")

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    },
    json={
        "model": "openrouter/free",
        "messages": [
            {
                "role": "user",
                "content": "Translate 'Hello, how are you?' into Tamil. Return only the translation."
            }
        ],
        "max_tokens": 500,
    },
    timeout=60,
)

print("HTTP status:", response.status_code)

if response.ok:
    data = response.json()
    print("Model:", data.get("model"))
    print("Message:", data["choices"][0]["message"])
    print("Finish reason:", data["choices"][0].get("finish_reason"))
    print("Usage:", data.get("usage"))
else:
    print("Error:", response.text)
