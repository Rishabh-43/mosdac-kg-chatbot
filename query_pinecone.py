import pinecone
from pinecone import ServerlessSpec

from sentence_transformers import SentenceTransformer

# Reconnect to your index
index_name = "mosdac-chat-index"
index = pinecone.Index(index_name)

# ✅ Initialize same embedding model used before
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Sample query
query_text = "What is monsoon?"
query_embedding = embed_model.encode(query_text).tolist()

# ✅ Query top 2 most relevant results
query_response = index.query(
    vector=query_embedding,
    top_k=2,
    include_metadata=True  # include this if you added metadata earlier
)

print("🔎 Query Results:")
for match in query_response["matches"]:
    print(f"ID: {match['id']}, Score: {match['score']}")