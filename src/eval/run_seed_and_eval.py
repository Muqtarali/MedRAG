from src.eval.evaluate_retrieval import load_qrels, load_queries, evaluate
from src.utils.vectorstore import VectorStore
from langchain.schema import Document
import json

# seed
vs = VectorStore()
sample_texts = [
    "Community-acquired pneumonia (CAP) in adults is commonly caused by Streptococcus pneumoniae.",
]
docs = [Document(page_content=t, metadata={"source": f"sample{i+1}"}) for i, t in enumerate(sample_texts)]
vs.add_documents(docs)
print(f"Seeded {len(docs)} docs into in-memory VectorStore")

# load queries and qrels
queries = load_queries('src/eval/queries.jsonl')
qrels = load_qrels('src/eval/qrels.tsv')
res = evaluate(vs, queries, qrels, k=5)
print(json.dumps(res['summary'], indent=2))
with open('src/eval/results.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=2)
print('Wrote src/eval/results.json')
