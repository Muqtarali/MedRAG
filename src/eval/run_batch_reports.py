"""Run a batch of queries through ReportGenerator, save each report, and compute
per-query and summary retrieval metrics. Produces JSON and CSV metric outputs.

Usage (from project root):
  python -m src.eval.run_batch_reports --queries src/eval/queries_multi.jsonl --qrels src/eval/qrels_multi.tsv --out-dir src/eval/batch_reports --k 5
"""
import argparse
import json
import os
import csv
from typing import List, Tuple

from src.report_generator import ReportGenerator
from src.eval.evaluate_retrieval import (
    load_qrels,
    load_queries,
    precision_at_k,
    recall_at_k,
    average_precision,
    ndcg_at_k,
    mrr,
)


def docid_from_doc(doc) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    docid = meta.get("source") or meta.get("doc_id") or meta.get("path") or meta.get("filename")
    if docid is None:
        docid = (getattr(doc, "page_content", "")[:30]).strip()
    return str(docid)


def run_batch(queries_path: str, qrels_path: str, out_dir: str, k: int = 5, use_web: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    reports_dir = os.path.join(out_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    qrels = load_qrels(qrels_path)
    queries = load_queries(queries_path)

    rg = ReportGenerator()
    # Ensure the vectorstore has documents. If empty, try seeding with demo documents.
    if rg.vs.is_empty():
        try:
            from src.eval import seed_vectorstore
            print("VectorStore empty — running seed_vectorstore.seed() to populate demo documents.")
            seed_vectorstore.seed()
            rg = ReportGenerator()  # re-init to pick up persisted store
        except Exception as e:
            print("Failed to seed vectorstore:", e)
    # If still empty, try to add the demo documents directly to the current vectorstore instance
    if rg.vs.is_empty():
        try:
            from langchain.schema import Document
            from src.eval.seed_vectorstore import sample_texts
            docs = [Document(page_content=t, metadata={"source": f"sample{i+1}"}) for i, t in enumerate(sample_texts)]
            rg.vs.add_documents(docs)
            print(f"Added {len(docs)} demo documents directly to VectorStore instance.")
        except Exception as e:
            print("Failed to directly add demo docs to vectorstore instance:", e)

    per_query = {}
    csv_rows: List[List] = []
    csv_header = ["qid", "ap", "ndcg", "mrr", "precision", "recall", "retrieved"]

    for qid, qtext in queries:
        print(f"Running qid={qid}: {qtext}")
        # try to pass minimal patient container (ReportGenerator expects a dict)
        patient = {"name": None}
        try:
            report, retrieved = rg.generate(patient, qtext, top_k=k, use_web=use_web, structured=False)
        except Exception as e:
            print(f"Error generating report for {qid}: {e}")
            report = str(e)
            retrieved = []

        # save report to file
        out_report_path = os.path.join(reports_dir, f"{qid}.json")
        with open(out_report_path, "w", encoding="utf-8") as f:
            json.dump({"qid": qid, "query": qtext, "report": report}, f, indent=2, ensure_ascii=False)

        # convert retrieved docs to ids
        retrieved_ids = [docid_from_doc(d) for d in retrieved]

        # compute metrics using qrels
        rels = qrels.get(qid, {})
        relevant_set = set(d for d, r in rels.items() if r > 0)

        ap = average_precision(retrieved_ids, relevant_set)
        ndcg = ndcg_at_k(retrieved_ids, rels, k)
        rr = mrr(retrieved_ids, relevant_set)
        prec = precision_at_k(retrieved_ids, relevant_set, k)
        rec = recall_at_k(retrieved_ids, relevant_set, k)

        per_query[qid] = {
            "query": qtext,
            "ap": ap,
            "ndcg": ndcg,
            "mrr": rr,
            "precision": prec,
            "recall": rec,
            "retrieved": retrieved_ids,
            "relevant": rels,
            "report_path": out_report_path,
        }

        csv_rows.append([qid, ap, ndcg, rr, prec, rec, ";".join(retrieved_ids)])

    # summary
    n = len(per_query)
    if n:
        summary = {
            "map": sum(v["ap"] for v in per_query.values()) / n,
            "mean_ndcg": sum(v["ndcg"] for v in per_query.values()) / n,
            "mean_mrr": sum(v["mrr"] for v in per_query.values()) / n,
            "mean_precision": sum(v["precision"] for v in per_query.values()) / n,
            "mean_recall": sum(v["recall"] for v in per_query.values()) / n,
        }
    else:
        summary = {"map": 0.0, "mean_ndcg": 0.0, "mean_mrr": 0.0, "mean_precision": 0.0, "mean_recall": 0.0}

    out_json = os.path.join(out_dir, "batch_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"per_query": per_query, "summary": summary}, f, indent=2, ensure_ascii=False)

    out_csv = os.path.join(out_dir, "batch_metrics.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)

    print("Wrote:", out_json, out_csv)
    print("Summary:", json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, help="Path to queries.jsonl")
    parser.add_argument("--qrels", required=True, help="Path to qrels.tsv")
    parser.add_argument("--out-dir", default="src/eval/batch_reports", help="Output directory")
    parser.add_argument("--k", type=int, default=5, help="k for @k metrics and retrieval")
    parser.add_argument("--use-web", action="store_true", help="Allow web evidence in report generation")
    args = parser.parse_args()

    run_batch(args.queries, args.qrels, args.out_dir, k=args.k, use_web=args.use_web)


if __name__ == "__main__":
    main()
