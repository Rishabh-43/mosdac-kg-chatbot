import pinecone
import os
from pinecone import ServerlessSpec


pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY") or "ZYNL2pcsk_7USieR_TWCp5dcfXBN4ePuRgXGuYCR2YDX8ipq2V43vc4My6mZZUn8VffNFoUgeN2",
    environment="us-east-1"
)

# ✅ STEP 1: Create the Pinecone Index
index_name = "mosdac-chat-index"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=768,  # Adjust based on your embedding size
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Index '{index_name}' created successfully!")
else:
    print(f"Index '{index_name}' already exists.")

# ✅ STEP 2: Connect to the index
index = pinecone.Index(index_name)
print(f"Connected to index: {index_name}")

from sentence_transformers import SentenceTransformer

# Initialize embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # Efficient & widely used
index_name = "mosdac-chat-index"
index = pinecone.Index(index_name)

# ✅ Example documents
texts = [
    "What is MOSDAC?",
    "How does rainfall measurement work?",
    "Tell me about Indian monsoon patterns.",
]

# ✅ Embed the texts
embeddings = embed_model.encode(texts).tolist()

# ✅ Prepare for upsert
vectors_to_upsert = [
    {"id": f"doc-{i}", "values": emb} for i, emb in enumerate(embeddings)
]

# ✅ Upsert to Pinecone
index.upsert(vectors=vectors_to_upsert)
print("Vectors upserted successfully!")