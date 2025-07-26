import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from googletrans import Translator
import google.generativeai as genai

# --------- GEMINI API KEY ----------
GEMINI_API_KEY = "AIzaSyAyhcpDKXBCs9J5B-lS_-RCbrCNcfKP7hw"
genai.configure(api_key=GEMINI_API_KEY)

# --------- Load Chroma DB ----------
@st.cache_resource
def load_vector_db():
    client = chromadb.PersistentClient(path="vector_db/chroma")
    collection = client.get_or_create_collection("mosdac_kg")
    return collection

# --------- Load Embedding Model ----------
@st.cache_resource
def load_model():
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# --------- Search from Vector DB ----------
def search(query, collection, model, k=5):
    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    texts = []
    for i in range(len(results["ids"][0])):
        text = results["metadatas"][0][i]["text"]
        source = results["metadatas"][0][i]["source"]
        texts.append((source, text))
    return texts

# --------- Generate Answer with Gemini ----------
def generate_answer_with_gemini(user_query, retrieved_data):
    if not retrieved_data:
        return "⚠ Sorry, I couldn't find any relevant information to answer your question."

    context_text = "\n".join([text for _, text in retrieved_data])
    prompt = f"""
    You are an intelligent assistant for ISRO's MOSDAC platform.
    Use the following context to answer the user query in simple terms:
    
    Context:
    {context_text}

    Question:
    {user_query}

    Respond in clear bullet points and avoid repeating the context. Be concise and specific.
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# --------- Streamlit UI ----------
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

    collection = load_vector_db()
    model = load_model()
    results = search(translated, collection, model)

    # 🔍 Optional: Debug - show matches
    with st.expander("🔍 Retrieved Context (Debug)"):
        for i, (src, txt) in enumerate(results, 1):
            st.markdown(f"**{i}. {src}**: {txt[:300]}...", unsafe_allow_html=True)

    with st.spinner("Generating answer..."):
        ai_answer = generate_answer_with_gemini(translated, results)

    st.markdown(f"<div class='message-container message-bot'>🤖 *Answer:* {ai_answer}</div>", unsafe_allow_html=True)
