import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from googletrans import Translator
import google.generativeai as genai

# ---------- CONFIGURE GEMINI API ----------
GEMINI_API_KEY = "AIzaSyAyhcpDKXBCs9J5B-lS_-RCbrCNcfKP7hw"
genai.configure(api_key=GEMINI_API_KEY)

# ---------- LOAD VECTOR DB ----------
@st.cache_resource
def load_vector_db():
    index = faiss.read_index("vector_db/faiss_index.bin")
    with open("vector_db/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# ---------- LOAD EMBEDDING MODEL ----------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ---------- SEARCH FUNCTION ----------
def search(query, index, metadata, model, k=10):
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = normalize(query_embedding, axis=1)
    D, I = index.search(query_embedding, k)
    results = [metadata[i] for i in I[0]]

    query_lower = query.lower()
    ranked = sorted(results, key=lambda x: query_lower in x[1].lower(), reverse=True)

    return ranked[:5]

# ---------- GEMINI ANSWER GENERATOR ----------
def generate_answer_with_gemini(user_query, retrieved_data):
    # Merge retrieved results into context
    context_text = "\n".join([text for _, text in retrieved_data])
    prompt = f"""
    You are an intelligent MOSDAC assistant.
    Use the following context to answer the user question:
    
    Context:
    {context_text}
    
    Question:
    {user_query}
    
    Provide a *clear, structured answer in bullet points*.
    Avoid just listing documents; summarize the information.
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# ---------- STREAMLIT UI ----------
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

query = st.text_input("Ask a question about missions, sensors, documents...")

if query:
    try:
        translated = translator.translate(query, src='auto', dest='en').text
    except Exception:
        translated = query
        st.warning("⚠ Translation failed, using original query.")

    st.markdown(f"<div class='message-container message-human'>👤 *You:* {query}</div>", unsafe_allow_html=True)

    index, metadata = load_vector_db()
    model = load_model()
    results = search(translated, index, metadata, model)

    # ✅ Instead of showing raw matches → Generate AI answer
    with st.spinner("Generating answer..."):
        ai_answer = generate_answer_with_gemini(translated, results)

    st.markdown(f"<div class='message-container message-bot'>🤖 *Answer:* {ai_answer}</div>", unsafe_allow_html=True)