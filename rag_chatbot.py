# rag_chatbot.py

import os
import streamlit as st
import pinecone  # ✅ updated import
from sentence_transformers import SentenceTransformer
from googletrans import Translator
import google.generativeai as genai

# ---------- CONFIG ---------- #
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_7USieR_TWCp5dcfXBN4ePuRgXGuYCR2YDX8ipq2V43vc4My6mZZUn8VffNFoUgeN2ZYNL2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAyhcpDKXBCs9J5B-lS_-RCbrCNcfKP7hw")
INDEX_NAME = "mosdac-rag"
EMBEDDING_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"

# ---------- API Setup ---------- #
genai.configure(api_key=GEMINI_API_KEY)

# ---------- Load Pinecone Index ---------- #
@st.cache_resource
def load_index():
    pinecone.init(api_key=PINECONE_API_KEY)
    return pinecone.Index(INDEX_NAME)

# ---------- Load Embedding Model ---------- #
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# ---------- Query Vector DB ---------- #
def search(query, index, model, k=5):
    query_vector = model.encode([query])[0].tolist()
    result = index.query(vector=query_vector, top_k=k, include_metadata=True)
    return [(match["metadata"]["source"], match["metadata"]["text"]) for match in result["matches"]]

# ---------- Gemini Response ---------- #
def generate_answer_with_gemini(question, retrieved_data):
    if not retrieved_data:
        return "⚠ Sorry, I couldn't find any relevant information for your question."

    context = "\n".join([text for _, text in retrieved_data])
    prompt = f"""
    You are a helpful assistant for ISRO's MOSDAC platform.
    Use the following context to answer the question clearly.

    Context:
    {context}

    Question:
    {question}

    Answer in bullet points. Be specific, avoid repetition, and provide only the necessary information.
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# ---------- Streamlit UI ---------- #
st.set_page_config(page_title="MOSDAC Chatbot", layout="centered")

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

# ---------- Query Handling ---------- #
query = st.text_input("Ask a question about missions, sensors, or documents...")

if query:
    try:
        translated_query = translator.translate(query, src="auto", dest="en").text
    except:
        translated_query = query
        st.warning("⚠ Translation failed, using original query.")

    st.markdown(f"<div class='message-container message-human'>👤 *You:* {query}</div>", unsafe_allow_html=True)

    index = load_index()
    model = load_model()
    results = search(translated_query, index, model)

    with st.expander("🔍 Retrieved Context"):
        for i, (src, txt) in enumerate(results, 1):
            st.markdown(f"**{i}. {src.upper()}**: {txt[:300]}...")

    with st.spinner("Generating answer..."):
        response = generate_answer_with_gemini(translated_query, results)

    st.markdown(f"<div class='message-container message-bot'>🤖 *Answer:* {response}</div>", unsafe_allow_html=True)
