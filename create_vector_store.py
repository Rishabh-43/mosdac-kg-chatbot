# create_vector_store.py

import pandas as pd
import faiss
import os
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import pickle

# Load all datasets
kg_df = pd.read_csv("kg_master.csv")
faq_df = pd.read_csv("mosdac_faq.csv")
docs_df = pd.read_csv("mosdac_docs.csv")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare texts
entries = []

# Add KG entries
for _, row in kg_df.iterrows():
    text = f"{row['source']} — {row['relation']} — {row['target']}"
    entries.append(("kg", text))

# Add FAQ entries
for _, row in faq_df.iterrows():
    q = str(row.get("question", ""))
    a = str(row.get("answer", ""))
    if q and a:
        entries.append(("faq", f"Q: {q} A: {a}"))

# Add document titles
for _, row in docs_df.iterrows():
    title = str(row.get("title", ""))
    url = str(row.get("url", ""))
    if title:
        entries.append(("doc", f"{title} — {url}"))

# Embed all entries
texts = [text for _, text in entries]
print(f"🔍 Embedding {len(texts)} entries…")
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = normalize(embeddings, axis=1)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index and metadata
os.makedirs("vector_db", exist_ok=True)
faiss.write_index(index, "vector_db/faiss_index.bin")

with open("vector_db/metadata.pkl", "wb") as f:
    pickle.dump(entries, f)

print("✅ Vector database created and saved in vector_db/")
