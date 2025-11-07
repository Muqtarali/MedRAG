from langchain.schema import Document
from src.utils.vectorstore import VectorStore

# Seed the vectorstore with one sample document used by run_report_test.py
sample_texts = [
    "Community-acquired pneumonia (CAP) in adults is commonly caused by Streptococcus pneumoniae.",
]

def seed():
    vs = VectorStore()
    if not vs.is_empty():
        print("VectorStore already has documents; seeding will append sample docs.")
    docs = [Document(page_content=t, metadata={"source": f"sample{i+1}"}) for i, t in enumerate(sample_texts)]
    vs.add_documents(docs)
    print(f"Seeded {len(docs)} documents into vectorstore at {vs.persist_path}")

if __name__ == '__main__':
    seed()
