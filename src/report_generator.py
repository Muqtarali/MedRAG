from typing import List, Tuple, Optional
import json
import datetime
import os
from src.utils.vectorstore import VectorStore
from src.prompts import build_report_prompt
from src.models.openai_client import call_openai_chat
import requests
import os
from src.utils.web_search import search_web
from langchain.schema import Document


class ReportGenerator:
    def __init__(self):
        self.vs = VectorStore()

    def retrieve(self, question: str, top_k: int = 6) -> List[Document]:
        results = self.vs.similarity_search_with_scores(question, k=top_k)
        # results is list of (Document, score)
        return [r[0] for r in results]

    def _render_markdown(self, structured: dict) -> str:
        """Render a formal report in Markdown from a structured dict."""
        lines = []
        title = structured.get("title") or "Clinical Report"
        lines.append(f"# {title}\n")

        meta = structured.get("meta", {})
        author = meta.get("author", "MedRAG Report Generator")
        date = meta.get("date", datetime.datetime.utcnow().isoformat())
        lines.append(f"**Prepared by:** {author}  \n**Date:** {date}\n")

        if structured.get("executive_summary"):
            lines.append("## Executive summary\n")
            lines.append(structured["executive_summary"].strip() + "\n")

        if structured.get("background"):
            lines.append("## Background\n")
            lines.append(structured["background"].strip() + "\n")

        if structured.get("methods"):
            lines.append("## Methods\n")
            lines.append(structured["methods"].strip() + "\n")

        if structured.get("findings"):
            lines.append("## Findings\n")
            if isinstance(structured["findings"], list):
                for f in structured["findings"]:
                    lines.append(f"- {f.strip()}")
            else:
                lines.append(structured["findings"].strip())
            lines.append("\n")

        if structured.get("recommendations"):
            lines.append("## Recommendations\n")
            if isinstance(structured["recommendations"], list):
                for r in structured["recommendations"]:
                    lines.append(f"- {r.strip()}")
            else:
                lines.append(structured["recommendations"].strip())
            lines.append("\n")

        if structured.get("references"):
            lines.append("## References\n")
            if isinstance(structured["references"], list):
                for ref in structured["references"]:
                    lines.append(f"- {ref}")
            else:
                lines.append(structured["references"]) 
            lines.append("\n")

        return "\n".join(lines)

    def generate(self, patient: dict, question: str, top_k: int = 6, llm_model: str = None, use_web: bool = False, output_path: Optional[str] = None, structured: bool = True) -> Tuple[str, List[Document]]:
        """Generate a formal clinical report.

        If `structured` is True, the generator asks the LLM to return JSON with keys:
        title, meta, executive_summary, background, methods, findings (list), recommendations (list), references (list).

        If `output_path` is provided, the report will be saved to that path (markdown).
        """
        if self.vs.is_empty():
            raise ValueError("Vector store is empty. Ingest documents first.")

        retrieved = self.retrieve(question, top_k=top_k)

        extra_context = ""
        if use_web:
            try:
                web_results = search_web(question, k=top_k)
                extra_context_lines = []
                for w in web_results:
                    extra_context_lines.append(f"{w.get('title','')}: {w.get('snippet','')} ({w.get('url','')})")
                extra_context = "\n".join(extra_context_lines)
            except Exception:
                extra_context = ""

        prompt = build_report_prompt(patient, retrieved, question, extra_context=extra_context)

        if structured:
            # Ask the model to return JSON structure for reliable parsing
            prompt = prompt + "\n\nOUTPUT FORMAT INSTRUCTIONS:\nReturn a single JSON object with the following keys: title, meta (subkeys: author,date), executive_summary, background, methods, findings (array of strings), recommendations (array of strings), references (array of strings). Respond only with valid JSON."

        # Call LLM (prefer OpenAI client; fallback to Groq HTTP API if OpenAI key not configured)
        llm_text = None
        try:
            llm_text = call_openai_chat(prompt, model=llm_model)
        except Exception:
            # Fallback to Groq if GROQ_API_KEY is present
            groq_url = os.environ.get("GROQ_API_URL") or "https://api.groq.com/openai/v1/chat/completions"
            groq_key = os.environ.get("GROQ_API_KEY")
            groq_model = os.environ.get("GROQ_MODEL") or llm_model or "gpt-4o-mini"
            if groq_key:
                headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
                payload = {"model": groq_model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 800}
                r = requests.post(groq_url, json=payload, headers=headers, timeout=120)
                try:
                    r.raise_for_status()
                    data = r.json()
                    # Prefer assistant content
                    try:
                        llm_text = data["choices"][0]["message"]["content"]
                    except Exception:
                        llm_text = data.get("choices", [{}])[0].get("text") or str(data)
                except Exception:
                    raise RuntimeError(f"Groq call failed: {r.status_code} {r.text}")
            else:
                raise

        report_text = llm_text
        if structured:
            # Try to extract JSON from the LLM response
            json_obj = None
            try:
                # Some models may wrap JSON in markdown, try to find the first '{'
                first_brace = llm_text.find('{')
                if first_brace != -1:
                    cleaned = llm_text[first_brace:]
                    json_obj = json.loads(cleaned)
            except Exception:
                json_obj = None

            if json_obj:
                report_text = self._render_markdown(json_obj)

        # Save to file if requested
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
            except Exception:
                pass

        return report_text, retrieved