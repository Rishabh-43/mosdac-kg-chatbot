import os
import pandas as pd
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ------------------ CONFIGURATION ------------------ #
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "pcsk_7USieR_TWCp5dcfXBN4ePuRgXGuYCR2YDX8ipq2V43vc4My6mZZUn8VffNFoUgeN2ZYNL2"
PINECONE_ENV = "us-east-1"
INDEX_NAME = "mosdac-rag"
EMBEDDING_DIM = 384
BATCH_SIZE = 100

# ------------------ LOAD DATA ------------------ #
kg_df = pd.read_csv("kg_master.csv")
faq_df = pd.read_csv("mosdac_faq.csv")
docs_df = pd.read_csv("mosdac_docs.csv")

entries = []

# Knowledge Graph
for _, row in kg_df.iterrows():
    text = f"{row['source']} — {row['relation']} — {row['target']}"
    entries.append({"text": text, "source": "kg"})

# FAQs
for _, row in faq_df.iterrows():
    q = str(row.get("question", "")).strip()
    a = str(row.get("answer", "")).strip()
    if q and a:
        entries.append({"text": f"Q: {q} A: {a}", "source": "faq"})

# Document Metadata
for _, row in docs_df.iterrows():
    title = str(row.get("title", "")).strip()
    url = str(row.get("url", "")).strip()
    if title:
        entries.append({"text": f"{title} — {url}", "source": "doc"})

# ------------------ LOAD EMBEDDING MODEL ------------------ #
print("🔍 Loading embedding model...")
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# ------------------ INIT PINECONE ------------------ #
print("🔗 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    print("📦 Creating Pinecone index...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
else:
    print("✅ Index already exists.")

index = pc.Index(INDEX_NAME)

# ------------------ UPLOAD EMBEDDINGS ------------------ #
print(f"🚀 Uploading {len(entries)} entries in batches of {BATCH_SIZE}...")

for i in tqdm(range(0, len(entries), BATCH_SIZE)):
    batch = entries[i:i+BATCH_SIZE]
    texts = [entry["text"] for entry in batch]
    embeddings = model.encode(texts).tolist()

    vectors = []
    for j, embedding in enumerate(embeddings):
        entry_id = f"vec-{i+j}"
        metadata = {
            "text": batch[j]["text"],
            "source": batch[j]["source"]
        }
        vectors.append({
            "id": entry_id,
            "values": embedding,
            "metadata": metadata
        })

    index.upsert(vectors=vectors)
    time.sleep(0.2)  # prevent throttling

print("✅ Vector data successfully uploaded to Pinecone.")
