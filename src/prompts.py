def build_report_prompt(patient: dict, retrieved_snippets: list, question: str, extra_context: str = "") -> str:
    """
    Composes a prompt for the LLM including patient info and retrieved evidence.
    """
    patient_section = []
    if patient:
        for k, v in patient.items():
            if v is not None and v != "":
                patient_section.append(f"{k}: {v}")
    patient_text = "\n".join(patient_section) if patient_section else "N/A"

    context_text = ""
    if retrieved_snippets:
        context_text = "\n\n".join([
            f"Source: {getattr(r, 'metadata', {}).get('source','unknown')}\nSnippet:\n{getattr(r, 'page_content','')}"
            for r in retrieved_snippets
        ])
    else:
        context_text = "No retrieved evidence."

    if extra_context:
        context_text = context_text + "\n\nAdditional web evidence:\n" + extra_context

    prompt = f"""
You are a medical assistant that prepares a clinical diagnostic report. Use the evidence provided from research papers, books and other medical sources. The final report should contain: 1) Brief history and summary 2) Differential diagnosis (top likely diagnoses with reasoning) 3) Recommended diagnostic tests and justification 4) Suggested immediate management or next steps 5) Short list of references (source names).

Patient:
{patient_text}

Question:
{question}

Evidence (use only what is appropriate, mark if evidence is uncertain):
{context_text}

Produce a structured diagnostic report in clear clinical language.
"""
    return prompt.strip()