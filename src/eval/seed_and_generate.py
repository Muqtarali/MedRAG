import json
from src.utils.vectorstore import VectorStore
from langchain.schema import Document
from src.eval.evaluate_retrieval import load_queries, load_qrels, evaluate


def docid_from_doc(doc) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    docid = meta.get("source") or meta.get("doc_id") or meta.get("path") or meta.get("filename")
    if docid is None:
        docid = (getattr(doc, "page_content", "")[:30]).strip()
    return str(docid)


def main():
    # seed
    vs = VectorStore()
    sample_texts = [
        "Community-acquired pneumonia (CAP) in adults is commonly caused by Streptococcus pneumoniae.",
    ]
    docs = [Document(page_content=t, metadata={"source": f"sample{i+1}"}) for i, t in enumerate(sample_texts)]
    vs.add_documents(docs)
    print(f"Seeded {len(docs)} documents into in-memory VectorStore")

    # generate candidates
    queries = load_queries('src/eval/queries.jsonl')
    cand_lines = []
    for qid, qtext in queries:
        hits = vs.similarity_search_with_scores(qtext, k=50)
        for rank, (doc, score) in enumerate(hits, start=1):
            docid = docid_from_doc(doc)
            cand_lines.append(f"{qid}\t{docid}\t{rank}\t{score}\n")
    out_cand = 'src/eval/candidates.tsv'
    with open(out_cand, 'w', encoding='utf-8') as f:
        f.writelines(cand_lines)
    print(f"Wrote {len(cand_lines)} candidate rows to {out_cand}")

    # evaluate
    qrels = load_qrels('src/eval/qrels.tsv')
    res = evaluate(vs, queries, qrels, k=5)
    out_eval = 'src/eval/results.json'
    with open(out_eval, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print('Evaluation summary:')
    print(json.dumps(res['summary'], indent=2))


if __name__ == '__main__':
    main()
