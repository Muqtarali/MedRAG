"""Attempt to call Groq with the Llama-4 model identifier you provided.

This script reads `GROQ_API_URL` and `GROQ_API_KEY` from the environment (or
from a local `.env`) and sends a single chat completion request. It prints the
JSON response. If your account doesn't have access to the model, you'll see a
model_not_found error (HTTP 404).
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_URL = os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_KEY:
    print("GROQ_API_KEY is not set in environment or .env")
    raise SystemExit(1)

MODEL = os.environ.get(
    "GROQ_MODEL",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
)

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Please provide a short summary of community-acquired pneumonia treatment."}],
    "temperature": 1,
    "max_completion_tokens": 512,
    "top_p": 1,
}

headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}

print(f"Sending request to Groq model: {MODEL} at {GROQ_URL}")

try:
    r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=120)
except Exception as e:
    print(f"Request error: {e}")
    raise

if r.status_code >= 200 and r.status_code < 300:
    # Print only the assistant text if present
    try:
        data = r.json()
    except Exception:
        print(r.text)
        raise SystemExit(0)

    assistant_text = None
    try:
        assistant_text = data["choices"][0]["message"]["content"]
    except Exception:
        # Try alternate shapes
        try:
            assistant_text = data["choices"][0]["text"]
        except Exception:
            assistant_text = None

    if assistant_text:
        print(assistant_text)
    else:
        # Fallback: pretty-print full JSON
        import json
        print(json.dumps(data, indent=2))
else:
    try:
        body = r.json()
    except Exception:
        body = r.text
    print(f"Groq API returned {r.status_code}: {body}")
