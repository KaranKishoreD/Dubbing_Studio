"""
Try every known Bodhan voice name to find which ones are valid.
Run:  python list_voices.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

client = OpenAI(
    base_url="https://api.bodhan.ai/v1",
    api_key=os.environ["BODHAN_SPEAK_KEY"],
)

# All known/suspected Bodhan voice names — probe each one
CANDIDATE_VOICES = [
    # Female (confirmed)
    "Kavya",
    "Anitha",
    "Sravani",
    "Deepika",
    "Lakshmi",
    "Ishita",
    "Anagha",
    "Dhara",
    "Kaur",
    "Itishree",
    "Prastuti",
    "Saba",
    "Bharati",
    "Srijana",
    # Male (suspected — need to verify)
    "Arjun",
    "Shaurya",
    "Vivaan",
    "Arun",
    "Krishna",
    "Karthik",
    "Ravi",
    "Suresh",
    "Amit",
    "Gurpreet",
    "Farhan",
    "Rajesh",
    "Mohan",
    "Vikram",
    "Rohan",
]

TEST_TEXT = "नमस्ते"  # short Hindi text — fast to synthesize

import time

print("Probing voices (valid = no error)...\n")
valid, invalid = [], []

for voice in CANDIDATE_VOICES:
    for attempt in range(3):
        try:
            resp = client.audio.speech.create(
                model="indic-speak",
                input=TEST_TEXT,
                voice=voice,
                extra_body={"instructions": json.dumps({"lang": "hi", "style": "happy"})},
            )
            if len(resp.content) > 500:
                valid.append(voice)
                print(f"  ✓ {voice}")
            else:
                invalid.append(voice)
                print(f"  ✗ {voice} (empty response)")
            break
        except Exception as e:
            msg = str(e)
            if "429" in msg:
                wait = 3 * (attempt + 1)
                print(f"  ⏳ {voice} rate limited, retrying in {wait}s...")
                time.sleep(wait)
                if attempt == 2:
                    # Assume valid if only rate limited, never 500'd
                    valid.append(voice)
                    print(f"  ✓ {voice} (assumed valid — only rate limited)")
            else:
                invalid.append(voice)
                print(f"  ✗ {voice}  ({msg[:70]})")
                break
    time.sleep(1.5)  # 1.5s between each voice to stay under rate limit

print(f"\n{'=' * 40}")
print(f"Valid voices ({len(valid)}): {valid}")
print(f"Invalid:     {invalid}")
