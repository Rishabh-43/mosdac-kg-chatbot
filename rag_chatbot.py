# rag_chatbot.py

import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from googletrans import Translator
import google.generativeai as genai
import os

# ---------- CONFIG ----------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "pcsk_7USieR_TWCp5dcfXBN4ePuRgXGuYCR2YDX8ipq2V43vc4My6mZZUn8VffNFoUgeN2ZYNL2"
PINECONE_ENV = "us-east-1"  # only needed for UI clarity, not used in code
INDEX_NAME = "mosdac-rag"
EMBEDDING_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"

GEMINI_API_KEY = "AIzaSyAyhcpDKXBCs9J5B-lS_-RCbrCNcfKP7hw"
genai.configure(api_key=GEMINI_API_KEY)

# ---------- Load Pinecone Client ----------
@st.cache_resource
def load_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    return index

# ---------- Load Embedding Model ----------
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# ---------- Search Function ----------
def search(query, index, model, k=5):
    query_embedding = model.encode([query])[0].tolist()
    response = index.query(vector=query_embedding, top_k=k, include_metadata=True)
    results = [(match['metadata']['source'], match['metadata']['text']) for match in response['matches']]
    return results

# ---------- Answer with Gemini ----------
def generate_answer_with_gemini(user_query, retrieved_data):
    if not retrieved_data:
        return "⚠ Sorry, I couldn't find any relevant information."

    context_text = "\n".join([text for _, text in retrieved_data])
    prompt = f"""
    You are an intelligent assistant for ISRO's MOSDAC platform.
    Use the following context to answer the user's question clearly and helpfully:

    Context:
    {context_text}

    Question:
    {user_query}

    Format the answer in bullet points if relevant. Be clear, concise, and avoid repeating the same content.
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="MOSDAC Chatbot", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        body, .stApp { background-color: #0f1117; color: #f0f2f6; }
        .stTextInput > div > div > input {
            background-color: #1c1e26; color: #f0f2f6;
            border: 1px solid #3a3a3a; border-radius: 10px; padding: 10px;
        }
        .message-container {
            padding: 1rem; margin-top: 1.5rem; border-radius: 10px;
            background-color: #1c1e26; border: 1px solid #333;
        }
        .message-human { color: #99c9ff; font-weight: 500; }
        .message-bot { color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

st.title("🛰 MOSDAC Knowledge Chatbot")
translator = Translator()

query = st.text_input("Ask a question about missions, sensors, or documents...")

if query:
    try:
        translated = translator.translate(query, src='auto', dest='en').text
    except Exception:
        translated = query
        st.warning("⚠ Translation failed, using original query.")

    st.markdown(f"<div class='message-container message-human'>👤 *You:* {query}</div>", unsafe_allow_html=True)

    index = load_pinecone_index()
    model = load_model()
    results = search(translated, index, model)

    with st.expander("🔍 Retrieved Context"):
        for i, (src, txt) in enumerate(results, 1):
            st.markdown(f"**{i}. {src}**: {txt[:300]}...")

    with st.spinner("Generating answer..."):
        answer = generate_answer_with_gemini(translated, results)

    st.markdown(f"<div class='message-container message-bot'>🤖 *Answer:* {answer}</div>", unsafe_allow_html=True)
