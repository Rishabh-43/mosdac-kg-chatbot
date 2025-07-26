# create_chroma_db.py

import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import tqdm

# Load data
kg_df = pd.read_csv("kg_master.csv")
faq_df = pd.read_csv("mosdac_faq.csv")
docs_df = pd.read_csv("mosdac_docs.csv")

# Load model
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# Collect entries: (source_type, text)
entries = []

for _, row in kg_df.iterrows():
    text = f"{row['source']} — {row['relation']} — {row['target']}"
    entries.append(("kg", text))

for _, row in faq_df.iterrows():
    q = str(row.get("question", "")).strip()
    a = str(row.get("answer", "")).strip()
    if q and a:
        entries.append(("faq", f"Q: {q} A: {a}"))

for _, row in docs_df.iterrows():
    title = str(row.get("title", "")).strip()
    url = str(row.get("url", "")).strip()
    if title:
        entries.append(("doc", f"{title} — {url}"))

# Embed and store in ChromaDB
texts = [entry[1] for entry in entries]
sources = [entry[0] for entry in entries]

print(f"🔍 Embedding {len(texts)} entries...")
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = normalize(embeddings, axis=1)

# Initialize Chroma
client = chromadb.PersistentClient(path="vector_db/chroma")
collection = client.get_or_create_collection("mosdac_kg")

# Insert into collection
for i, (text, source) in enumerate(zip(texts, sources)):
    collection.add(
        ids=[f"entry-{i}"],
        embeddings=[embeddings[i].tolist()],
        metadatas=[{"text": text, "source": source}]
    )

print("✅ ChromaDB vector store created at vector_db/chroma/")
