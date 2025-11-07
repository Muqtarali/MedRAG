"""
Small CLI to ingest files and run a sample query without using the HTTP API.
"""
import sys
from src.ingest import ingest_files
from src.report_generator import ReportGenerator

def demo_ingest(paths):
    res = ingest_files(paths)
    print("Ingested:", res)

def demo_query(patient, question):
    gen = ReportGenerator()
    report, retrieved = gen.generate(patient=patient, question=question)
    print("=== Report ===")
    print(report)
    print("\n=== Retrieved snippets ===")
    for i, r in enumerate(retrieved):
        print(f"[{i}] source: {r.metadata.get('source')} snippet: {r.page_content[:300]}...\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.cli_demo ingest file1.pdf file2.pdf")
        print("   or: python -m src.cli_demo query")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "ingest":
        paths = sys.argv[2:]
        demo_ingest(paths)
    elif cmd == "query":
        patient = {"name": "Test Patient", "age": 60, "sex": "F", "history": "Progressive dyspnea"}
        question = "Generate a diagnostic report for progressive dyspnea"
        demo_query(patient, question)
    else:
        print("Unknown command")