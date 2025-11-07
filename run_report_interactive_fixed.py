"""Interactive or non-interactive CLI to accept patient info and a question,
run the RAG report generator, show retrieved evidence, and optionally save the
query for later evaluation.

Usage:
    Interactive: python run_report_interactive_fixed.py
    Non-interactive: python run_report_interactive_fixed.py --question "..." --name "..." --age 22 --sex M --history "..." --top-k 6 --use-web
"""
import argparse
import json
import uuid
from datetime import datetime
from src.report_generator import ReportGenerator

EVAL_QUERIES = "src/eval/queries.jsonl"


def prompt_patient():
    print("Enter patient information (leave blank to skip)")
    name = input("Name: ").strip()
    age = input("Age: ").strip()
    sex = input("Sex: ").strip()
    history = input("History / brief clinical background: ").strip()
    return {
        "name": name or None,
        "age": int(age) if age.isdigit() else None,
        "sex": sex or None,
        "history": history or None,
    }


def save_query(qid: str, query: str):
    # append to queries.jsonl
    entry = {"qid": qid, "query": query}
    with open(EVAL_QUERIES, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_report(patient: dict, question: str, use_web: bool = False, top_k: int = 6, structured: bool = False):
    rg = ReportGenerator()
    print("Running report generator... (this may take a few seconds)")
    report, retrieved = rg.generate(patient, question, top_k=top_k, use_web=use_web, structured=structured)

    print('\n===== GENERATED REPORT =====')
    print(report)
    print('===== END REPORT =====\n')

    print('Retrieved evidence:')
    for i, doc in enumerate(retrieved, start=1):
        meta = getattr(doc, 'metadata', {}) or {}
        src = meta.get('source') or meta.get('doc_id') or meta.get('path') or 'unknown'
        snippet = getattr(doc, 'page_content', '')[:300].replace('\n', ' ')
        print(f"{i}. [{src}] score={meta.get('score','?')} -> {snippet}")

    return report, retrieved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Patient name")
    parser.add_argument("--age", type=int, help="Patient age")
    parser.add_argument("--sex", help="Patient sex")
    parser.add_argument("--history", help="Patient brief history")
    parser.add_argument("--question", help="Clinical question to ask")
    parser.add_argument("--use-web", action="store_true", help="Include web evidence")
    parser.add_argument("--top-k", type=int, default=6, help="Number of retrieval hits to include")
    parser.add_argument("--structured", action="store_true", help="Request structured JSON output from LLM")
    parser.add_argument("--save-query", action="store_true", help="Save the query to eval/queries.jsonl for later labeling")
    args = parser.parse_args()

    if args.question:
        # non-interactive mode
        patient = {"name": args.name, "age": args.age, "sex": args.sex, "history": args.history}
        question = args.question
        report, retrieved = run_report(patient, question, use_web=args.use_web, top_k=args.top_k, structured=args.structured)
        if args.save_query:
            qid = f"iq-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
            save_query(qid, question)
            print(f"Saved query as qid={qid} -> {EVAL_QUERIES}")
        return

    # interactive mode
    print("RAG Interactive Report CLI")
    patient = prompt_patient()
    question = input("Enter clinical question to ask the system: ").strip()
    if not question:
        print("No question entered. Exiting.")
        return

    use_web = input("Include web evidence? (y/N): ").strip().lower() == "y"
    top_k_input = input("How many retrieval hits to include (default 6): ").strip()
    try:
        top_k = int(top_k_input) if top_k_input else 6
    except Exception:
        top_k = 6

    structured = input("Request structured JSON output? (y/N): ").strip().lower() == "y"

    report, retrieved = run_report(patient, question, use_web=use_web, top_k=top_k, structured=structured)

    # save query for later evaluation
    qid = f"iq-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    save_it = input("Save this query to eval/queries.jsonl for later labeling? (Y/n): ").strip().lower()
    if save_it in ("", "y", "yes"):
        save_query(qid, question)
        print(f"Saved query as qid={qid} -> {EVAL_QUERIES}")
    else:
        print("Query not saved.")


if __name__ == '__main__':
    main()
