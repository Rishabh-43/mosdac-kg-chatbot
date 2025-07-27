import pinecone
from sentence_transformers import SentenceTransformer
import os

# Load environment variables (optional) or directly paste your API key
PINECONE_API_KEY = "pcsk_7USieR_TWCp5dcfXBN4ePuRgXGuYCR2YDX8ipq2V43vc4My6mZZUn8VffNFoUgeN2ZYNL2"
PINECONE_ENV = "us-east-1-aws"  # e.g., "gcp-starter" or "us-central1-gcp"
INDEX_NAME = "mosdac-rag"  # your index name

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)

# Load embedding model
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

def embed_and_upsert(texts, namespace="default"):
    vectors = model.encode(texts, show_progress_bar=True)
    ids = [f"id-{i}" for i in range(len(texts))]
    index.upsert(vectors=zip(ids, vectors), namespace=namespace)
    return ids

def query_pinecone(query, top_k=5, namespace="default"):
    vector = model.encode([query])[0]
    results = index.query(vector=vector.tolist(), top_k=top_k, include_metadata=True, namespace=namespace)
    return results
