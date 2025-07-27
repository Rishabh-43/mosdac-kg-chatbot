from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pinecone
from sentence_transformers import SentenceTransformer

# 🔧 FastAPI App Initialization
app = FastAPI(
    title="Pinecone Chatbot API",
    description="Semantic chatbot powered by sentence transformers and Pinecone",
    version="1.0"
)

# 🛡️ CORS Settings (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your domain for stricter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🌲 Pinecone Initialization
pinecone.init(
    api_key="your-api-key",           # 🔑 Replace with your Pinecone API key
    environment="your-environment"    # 🌍 Example: "gcp-starter", "us-west1-gcp"
)

# 🔎 Create Index Object
index = pinecone.Index("your-index-name")  # 🔗 Replace with your actual index name

# 🔠 Load Embedding Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # ⏩ Lightweight & fast

# 🧾 Input Schema
class ChatQuery(BaseModel):
    query: str

# 💬 Chat Endpoint
@app.post("/chat")
async def chat_endpoint(query: ChatQuery):
    query_text = query.query
    query_vector = embedder.encode(query_text).tolist()
    
    search_result = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    # 📚 Extract Answers
    answers = [match["metadata"]["text"] for match in search_result["matches"]]
    return {"question": query_text, "answer": answers}

# 🏠 Root Endpoint
@app.get("/")
def root():
    return {"message": "🚀 Pinecone Chatbot API is running!"}