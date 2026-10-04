import os
import pickle
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss

print("Loading FinanceBench...")
ds = load_dataset("PatronusAI/financebench")["train"]

documents = []

for row in ds:
    for ev in row["evidence"]:
        text = ev["evidence_text"]
        page = ev["evidence_page_num"]
        doc_name = ev["doc_name"]

        if text and len(text.strip()) > 50:
            documents.append({
                "text": text,
                "doc_name": doc_name,
                "page": page,
            })

print(f"Evidence chunks collected: {len(documents)}")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [d["text"] for d in documents]

print("Creating embeddings...")
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

os.makedirs("vectorstore", exist_ok=True)

faiss.write_index(index, "vectorstore/financebench.index")

with open("vectorstore/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print()
print("================================")
print("INDEX CREATED SUCCESSFULLY!")
print(f"Chunks: {len(documents)}")
print("Saved in: vectorstore/")
print("================================")