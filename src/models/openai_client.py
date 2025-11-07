import os
from typing import Optional
from src.config import settings
import openai

# Ensure the API key is set in environment before creating client
if settings.OPENAI_API_KEY:
    openai.api_key = settings.OPENAI_API_KEY

def call_openai_chat(prompt: str, model: str = None, temperature: float = 0.2, max_tokens: int = 800) -> str:
    """
    Calls OpenAI ChatCompletion (chat-based LLM). Requires OPENAI_API_KEY in env.
    """
    model = model or settings.LLM_MODEL
    # Basic safety: if no API key set, raise
    if not getattr(openai, "api_key", None):
        raise RuntimeError("OpenAI API key not configured. Set OPENAI_API_KEY in environment to use OpenAI LLM.")
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role":"system","content":"You are a concise, clinically-aware assistant that writes diagnostic reports."},
            {"role":"user","content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content