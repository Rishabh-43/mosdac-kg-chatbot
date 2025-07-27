from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "🚀 Test API is running!"}




from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

class ChatQuery(BaseModel):
    query: str

@app.post("/chat")
async def chat(query: ChatQuery):
    # Replace this stub with your Pinecone logic
    response = {"answer": f"You said: {query.query}"}
    return response