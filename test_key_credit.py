
import os
import requests

key = os.environ.get("OPENROUTER_API_KEY")
if not key:
    raise RuntimeError("Set OPENROUTER_API_KEY first.")

response = requests.get(
    "https://openrouter.ai/api/v1/key",
    headers={"Authorization": f"Bearer {key}"},
    timeout=30,
)

print("HTTP status:", response.status_code)

if response.ok:
    data = response.json()["data"]

    print("Key label:", data.get("label"))
    print("Key spending limit:", data.get("limit"))
    print("Remaining key limit:", data.get("limit_remaining"))
    print("Total usage:", data.get("usage"))
    print("Today's usage:", data.get("usage_daily"))
    print("This month's usage:", data.get("usage_monthly"))
    print("Free requests remaining today:",
          data.get("free_model_daily_requests", {}).get("remaining"))
else:
    print("Error:", response.text)
