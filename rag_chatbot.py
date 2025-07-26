import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from googletrans import Translator

# Load vector DB
@st.cache_resource
def load_vector_db():
    index = faiss.read_index("vector_db/faiss_index.bin")
    with open("vector_db/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Improved Search function with relevance filtering
def search(query, index, metadata, model, k=10):
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = normalize(query_embedding, axis=1)
    D, I = index.search(query_embedding, k)
    results = [metadata[i] for i in I[0]]

    # Prioritize entries that contain the query term directly
    query_lower = query.lower()
    ranked = sorted(results, key=lambda x: query_lower in x[1].lower(), reverse=True)

    return ranked[:5]


# Streamlit Config
st.set_page_config(page_title="MOSDAC Chatbot", layout="centered", initial_sidebar_state="collapsed")

# Custom dark theme CSS
st.markdown("""
    <style>
        body, .stApp {
            background-color: #0f1117;
            color: #f0f2f6;
        }
        .stTextInput > div > div > input {
            background-color: #1c1e26;
            color: #f0f2f6;
            border: 1px solid #3a3a3a;
            border-radius: 10px;
            padding: 10px;
        }
        .message-container {
            padding: 1rem;
            margin-top: 1.5rem;
            border-radius: 10px;
            background-color: #1c1e26;
            border: 1px solid #333;
        }
        .message-human {
            color: #99c9ff;
            font-weight: 500;
        }
        .message-bot {
            color: #ffffff;
        }
        .result-type {
            color: #bbbbbb;
            font-size: 0.8rem;
            margin-top: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🛰️ MOSDAC Knowledge Chatbot")
translator = Translator()

# User Query Input
query = st.text_input("Ask a question about missions, sensors, documents...")

if query:
    try:
        translated = translator.translate(query, src='auto', dest='en').text
    except Exception:
        translated = query
        st.warning("⚠️ Translation failed, using original query.")

    st.markdown(f"<div class='message-container message-human'>👤 **You:** {query}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-type'>🌐 Translated: {translated}</div>", unsafe_allow_html=True)

    # Load and search
    index, metadata = load_vector_db()
    model = load_model()
    results = search(translated, index, metadata, model)

    st.markdown("### 🤖 Top Matches")
    for source_type, text in results:
        if source_type == "faq":
            st.markdown(f"<div class='message-container message-bot'>📘 **FAQ**: {text}</div>", unsafe_allow_html=True)
        elif source_type == "doc":
            try:
                title, url = text.split(" — ")
                st.markdown(f"<div class='message-container message-bot'>📄 **Document**: <a href='{url}' target='_blank' style='color:#66ccff'>{title}</a></div>", unsafe_allow_html=True)
            except:
                st.markdown(f"<div class='message-container message-bot'>📄 **Document**: {text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='message-container message-bot'>🧠 **KG Entry**: {text}</div>", unsafe_allow_html=True)
