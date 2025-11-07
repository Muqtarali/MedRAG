import argparse
import json
from typing import List, Tuple

from src.eval.evaluate_retrieval import load_queries
from src.utils.vectorstore import VectorStore


def docid_from_doc(doc) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    docid = meta.get("source") or meta.get("doc_id") or meta.get("path") or meta.get("filename")
    if docid is None:
        docid = (getattr(doc, "page_content", "")[:30]).strip()
    return str(docid)


def generate(queries_path: str, out_path: str, top_n: int = 50):
    queries = load_queries(queries_path)
    vs = VectorStore()
    if vs.is_empty():
        print("Warning: vectorstore empty. Seed or ingest documents first.")

    lines = []
    for qid, qtext in queries:
        hits = vs.similarity_search_with_scores(qtext, k=top_n)
        for rank, (doc, score) in enumerate(hits, start=1):
            docid = docid_from_doc(doc)
            # TSV: qid \t docid \t rank \t score
            lines.append(f"{qid}\t{docid}\t{rank}\t{score}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Wrote {len(lines)} candidate rows to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out", default="src/eval/candidates.tsv")
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()
    generate(args.queries, args.out, top_n=args.top)


if __name__ == '__main__':
    main()
