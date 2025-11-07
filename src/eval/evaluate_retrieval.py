import argparse
import json
from typing import Dict, List, Tuple
import math

from src.utils.vectorstore import VectorStore


def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    """Load qrels from a TSV: qid\tdocid\trelevance (int)."""
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, docid, rel = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[docid] = rel
    return qrels


def load_queries(path: str) -> List[Tuple[str, str]]:
    """Load queries from a JSONL or JSON file.
    JSONL: each line is {"qid": "q1", "query": "text"}
    JSON: [{"qid":"q1","query":"..."}, ...]
    """
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        # detect json vs jsonl by first non-whitespace char
        if first.strip().startswith("["):
            data = json.load(f)
            for item in data:
                queries.append((str(item["qid"]), item["query"]))
        else:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                queries.append((str(item["qid"]), item["query"]))
    return queries


def precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    return sum(1 for d in retrieved_k if d in relevant) / float(len(retrieved_k))


def recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    retrieved_k = retrieved[:k]
    if not relevant:
        return 0.0
    return sum(1 for d in retrieved_k if d in relevant) / float(len(relevant))


def average_precision(retrieved: List[str], relevant: set) -> float:
    # AP: average of precision@i over relevant doc positions
    num_relevant = 0
    score = 0.0
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            num_relevant += 1
            score += num_relevant / i
    if num_relevant == 0:
        return 0.0
    return score / num_relevant


def dcg_at_k(retrieved: List[str], rel_dict: Dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, d in enumerate(retrieved[:k], start=1):
        rel = rel_dict.get(d, 0)
        if i == 1:
            dcg += rel
        else:
            dcg += rel / math.log2(i)
    return dcg


def ideal_dcg(rel_scores: List[int], k: int) -> float:
    # sort descending
    sorted_rels = sorted(rel_scores, reverse=True)
    return dcg_at_k([str(i) for i in range(len(sorted_rels))], {str(i): sorted_rels[i] for i in range(len(sorted_rels))}, k)


def ndcg_at_k(retrieved: List[str], rel_dict: Dict[str, int], k: int) -> float:
    dcg = dcg_at_k(retrieved, rel_dict, k)
    rels = list(rel_dict.values())
    if not rels:
        return 0.0
    ideal = ideal_dcg(rels, k)
    if ideal == 0:
        return 0.0
    return dcg / ideal


def mrr(retrieved: List[str], relevant: set) -> float:
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def evaluate(vs: VectorStore, queries: List[Tuple[str, str]], qrels: Dict[str, Dict[str, int]], k: int = 10) -> Dict:
    results = {}
    sum_ap = 0.0
    sum_ndcg = 0.0
    sum_mrr = 0.0
    sum_prec = 0.0
    sum_recall = 0.0
    n = 0
    for qid, qtext in queries:
        # run retrieval
        hits = vs.similarity_search_with_scores(qtext, k=k)
        retrieved_ids = []
        for doc, score in hits:
            meta = getattr(doc, "metadata", {}) or {}
            docid = meta.get("source") or meta.get("doc_id") or meta.get("path") or meta.get("filename")
            if docid is None:
                # fallback to truncated content as id (not ideal)
                docid = (getattr(doc, "page_content", "")[:30]).strip()
            retrieved_ids.append(str(docid))

        rels = qrels.get(qid, {})
        relevant_set = set(d for d, r in rels.items() if r > 0)

        ap = average_precision(retrieved_ids, relevant_set)
        ndcg = ndcg_at_k(retrieved_ids, rels, k)
        rr = mrr(retrieved_ids, relevant_set)
        prec = precision_at_k(retrieved_ids, relevant_set, k)
        rec = recall_at_k(retrieved_ids, relevant_set, k)

        results[qid] = {
            "ap": ap,
            "ndcg": ndcg,
            "mrr": rr,
            "precision": prec,
            "recall": rec,
            "retrieved": retrieved_ids,
            "relevant": rels,
        }
        sum_ap += ap
        sum_ndcg += ndcg
        sum_mrr += rr
        sum_prec += prec
        sum_recall += rec
        n += 1

    summary = {
        "map": (sum_ap / n) if n else 0.0,
        "mean_ndcg": (sum_ndcg / n) if n else 0.0,
        "mean_mrr": (sum_mrr / n) if n else 0.0,
        "mean_precision": (sum_prec / n) if n else 0.0,
        "mean_recall": (sum_recall / n) if n else 0.0,
    }
    return {"per_query": results, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, help="Path to queries.jsonl or queries.json")
    parser.add_argument("--qrels", required=True, help="Path to qrels.tsv (qid\tdocid\trelevance)")
    parser.add_argument("--k", type=int, default=10, help="k for @k metrics")
    parser.add_argument("--out", default="eval_results.json", help="Output JSON file")
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    queries = load_queries(args.queries)

    vs = VectorStore()
    if vs.is_empty():
        print("Warning: vectorstore is empty. Run ingestion first or add sample docs.")

    res = evaluate(vs, queries, qrels, k=args.k)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
