import os
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ✅ Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "your_api_key_here"
INDEX_NAME = "mosdac-chat-index"

# ✅ Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # for all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# ✅ Connect to index
index = pc.Index(INDEX_NAME)

# ✅ Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ FastAPI setup
app = FastAPI()

# ✅ Request schema
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

# ✅ Embedding function
def get_embedding(text: str):
    return embed_model.encode(text).tolist()

# ✅ Query endpoint
@app.post("/query")
async def query_index(request: QueryRequest):
    query_embedding = get_embedding(request.query)
    result = index.query(vector=query_embedding, top_k=request.top_k, include_metadata=True)
    matches = result.get("matches", [])
    responses = [match["metadata"].get("text", f"ID: {match['id']}") for match in matches]
    return {"responses": responses or ["Sorry, no relevant match found."]}

# ✅ Bulk upload endpoint (optional)
@app.post("/upload")
async def upload_data():
    data = pd.read_csv("mosdac_data.csv")
    to_upsert = []
    for idx, row in data.iterrows():
        text = row["text"]
        embedding = get_embedding(text)
        to_upsert.append((str(idx), embedding, {"text": text}))
    index.upsert(vectors=to_upsert)
    return {"status": "✅ Upload completed"}

# ✅ Health check
@app.get("/")
def root():
    return {"message": "🚀 Pinecone Chatbot API is running!"}