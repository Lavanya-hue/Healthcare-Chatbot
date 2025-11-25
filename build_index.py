# build_index.py
import os
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DATA_FILE = "knowledge/health_faq.txt"
INDEX_PATH = "index.faiss"
META_PATH = "index_meta.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_docs(data_file):
    with open(data_file, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    # Split by double newline for simple chunking
    chunks = [chunk.strip() for chunk in raw.split("\n\n") if chunk.strip()]
    return chunks

def main():
    docs = load_docs(DATA_FILE)
    print(f"Loaded {len(docs)} doc chunks.")

    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(docs, show_progress_bar=True, convert_to_numpy=True)

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

    meta = {"docs": docs}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved index to {INDEX_PATH} and metadata to {META_PATH}.")

if __name__ == "__main__":
    main()
