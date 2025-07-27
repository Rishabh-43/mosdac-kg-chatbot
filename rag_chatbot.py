import os
import streamlit as st
import pinecone
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from googletrans import Translator
import google.generativeai as genai
from typing import List, Tuple, Optional

# ========== CONFIGURATION ========== #
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
INDEX_NAME = "mosdac-rag"
EMBEDDING_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
PINECONE_ENV = "us-east1-gcp"

# ========== INITIALIZATION ========== #
@st.cache_resource
def initialize_services() -> Tuple[Optional[pinecone.Index], Optional[SentenceTransformer]]:
    """Initialize Pinecone and SentenceTransformer with error handling"""
    try:
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if INDEX_NAME not in pc.list_indexes().names():
            st.error(f"Index '{INDEX_NAME}' not found. Available indexes: {pc.list_indexes().names()}")
            return None, None
        
        # Initialize Sentence Transformer
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return pc.Index(INDEX_NAME), model
    
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return None, None

# ========== CORE FUNCTIONS ========== #
def query_vector_db(query: str, 
                   index: pinecone.Index, 
                   model: SentenceTransformer, 
                   top_k: int = 5) -> List[Tuple[str, str]]:
    """Query Pinecone vector database"""
    try:
        query_embedding = model.encode(query).tolist()
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [(match["metadata"]["source"], match["metadata"]["text"]) for match in results["matches"]]
    except Exception as e:
        st.error(f"Vector search failed: {str(e)}")
        return []

def generate_response(question: str, context_data: List[Tuple[str, str]]) -> str:
    """Generate response using Gemini"""
    if not context_data:
        return "⚠ No relevant information found. Please try rephrasing your question."
    
    context = "\n".join([f"Source: {src}\nContent: {text}" for src, text in context_data])
    
    prompt = f"""You are an expert assistant for ISRO's MOSDAC platform. 
Answer the question using ONLY the provided context. Be specific and concise.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (in markdown with bullet points when appropriate):"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠ Error generating answer: {str(e)}"

# ========== STREAMLIT UI ========== #
st.set_page_config(
    page_title="MOSDAC Knowledge Chatbot",
    page_icon="🛰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0f1117;
        color: #f0f2f6;
    }
    .stTextInput input {
        background-color: #1c1e26 !important;
        color: white !important;
        border-radius: 10px;
        padding: 12px;
    }
    .message-user {
        background-color: #1c1e26;
        border-left: 4px solid #4e8cff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .message-bot {
        background-color: #1c1e26;
        border-left: 4px solid #00c897;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .stSpinner > div {
        color: #00c897 !important;
    }
</style>
""", unsafe_allow_html=True)

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Main Interface
st.title("🛰 MOSDAC Knowledge Chatbot")
st.caption("Ask about ISRO missions, sensors, or documentation")

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="message-user">
            <strong>👤 You:</strong> {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message-bot">
            <strong>🤖 Assistant:</strong> {message["content"]}
        </div>
        """, unsafe_allow_html=True)

# Chat input
user_query = st.chat_input("Type your question here...")

if user_query:
    # Add user query to history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.rerun()
    
    try:
        # Initialize services
        index, model = initialize_services()
        
        if index and model:
            # Translate query
            translator = Translator()
            try:
                translated_query = translator.translate(user_query, dest="en").text
            except:
                translated_query = user_query
                st.warning("Translation service unavailable, using original query")
            
            # Vector search
            with st.spinner("🔍 Searching knowledge base..."):
                search_results = query_vector_db(translated_query, index, model)
                
                # Show context in expander
                with st.expander("View retrieved context", expanded=False):
                    for i, (source, text) in enumerate(search_results, 1):
                        st.markdown(f"""
                        **Match {i}**  
                        **Source:** `{source}`  
                        **Content:** {text[:250]}...
                        """)
            
            # Generate response
            with st.spinner("💡 Generating answer..."):
                bot_response = generate_response(translated_query, search_results)
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                st.rerun()
    
    except Exception as e:
        st.error(f"System error: {str(e)}")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "⚠ Sorry, I encountered an error processing your request."
        })
        st.rerun()