import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

from pinecone import Pinecone, ServerlessSpec

# ---------- CONFIG ----------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "pcsk_7USieR_TWCp5dcfXBN4ePuRgXGuYCR2YDX8ipq2V43vc4My6mZZUn8VffNFoUgeN2ZYNL2"
INDEX_NAME = "mosdac-rag"
DIMENSION = 384

# ---------- Load Data ----------
kg_df = pd.read_csv("kg_master.csv")
faq_df = pd.read_csv("mosdac_faq.csv")
docs_df = pd.read_csv("mosdac_docs.csv")

entries = []

# Combine KG entries
for _, row in kg_df.iterrows():
    text = f"{row['source']} — {row['relation']} — {row['target']}"
    entries.append({"text": text, "source": "kg"})

# Combine FAQ entries
for _, row in faq_df.iterrows():
    q = str(row.get("question", "")).strip()
    a = str(row.get("answer", "")).strip()
    if q and a:
        entries.append({"text": f"Q: {q} A: {a}", "source": "faq"})

# Combine Document entries
for _, row in docs_df.iterrows():
    title = str(row.get("title", "")).strip()
    url = str(row.get("url", "")).strip()
    if title:
        entries.append({"text": f"{title} — {url}", "source": "doc"})

# ---------- Init Embedding Model ----------
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# ---------- Init Pinecone ----------
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)

# ---------- Upload Embeddings ----------
batch_size = 100

print(f"🚀 Embedding and uploading {len(entries)} entries...")

for i in tqdm(range(0, len(entries), batch_size)):
    batch = entries[i:i+batch_size]
    texts = [e["text"] for e in batch]
    embeddings = model.encode(texts).tolist()

    to_upsert = []
    for j, emb in enumerate(embeddings):
        entry_id = f"vec-{i+j}"
        metadata = {
            "text": batch[j]["text"],
            "source": batch[j]["source"]
        }
        to_upsert.append({"id": entry_id, "values": emb, "metadata": metadata})

    index.upsert(vectors=to_upsert)
    time.sleep(0.2)  # Avoid rate limits

print("✅ All entries uploaded to Pinecone index.")
