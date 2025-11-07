"""Minimal Groq runner that calls the configured Groq/OpenAI-compatible endpoint.

This script uses `src.models.openai_client.chat_completion` to send a prompt built
from a small set of sample texts. Set `GROQ_API_URL` and `GROQ_API_KEY` in the
environment before running.
"""
import os
import requests
import textwrap
from dotenv import load_dotenv

# Load local .env so users can store keys in project root
load_dotenv()

SAMPLE_TEXTS = [
    "Community-acquired pneumonia (CAP) in adults is commonly caused by Streptococcus pneumoniae.",
    "Recommended empiric antibiotics include a macrolide (e.g., azithromycin) or doxycycline for otherwise healthy outpatients.",
    "For hospitalized patients with severe disease, broader coverage or combination therapy may be indicated.",
]


def build_context(texts):
    return "\n\n---\n".join([f"Source {i}: {t}" for i, t in enumerate(texts)])


def main():
    groq_url = os.environ.get("GROQ_API_URL")
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_url or not groq_key:
        print("GROQ_API_URL and GROQ_API_KEY must be set in the environment")
        return

    query = "What is the recommended empiric outpatient antibiotic for community-acquired pneumonia?"
    context = build_context(SAMPLE_TEXTS)
    prompt = textwrap.dedent(f"""
    You are a medical assistant. Answer concisely using the context below and cite source ids.

    Question: {query}

    Context:
    {context}

    Answer:
    """)

    # Allow a single model override or try a short candidate list
    env_model = os.environ.get("GROQ_MODEL")
    candidates = [env_model] if env_model else [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3o",
        "gpt-3o-mini",
        "gpt-2o",
    ]

    headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
    last_err = None

    for model in candidates:
        if not model:
            continue
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        print(f"🧠 Trying Groq model: {model}...")
        try:
            r = requests.post(groq_url, json=payload, headers=headers, timeout=60)
        except Exception as e:
            print(f"Request failed for model {model}: {e}")
            last_err = str(e)
            continue

        if r.status_code >= 200 and r.status_code < 300:
            try:
                data = r.json()
            except Exception:
                print("Groq response (non-json):")
                print(r.text)
                return
            print("Groq response JSON:")
            print(data)
            return

        # non-2xx -> record and continue
        try:
            body = r.json()
        except Exception:
            body = r.text
        print(f"Groq API returned {r.status_code} for model {model}: {body}")
        last_err = body

    print("All candidate models failed. Last error:")
    print(last_err)


if __name__ == '__main__':
    main()
