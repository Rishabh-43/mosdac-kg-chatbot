# First try importing critical dependencies with proper error handling
try:
    import torch  # Must be imported before sentence-transformers
    from sentence_transformers import SentenceTransformer
    import scipy
    import nltk
except ImportError as e:
    raise ImportError(
        "Critical dependency error. Please run:\n"
        "pip install torch==2.0.1 sentence-transformers==2.2.2 scipy==1.10.1 nltk==3.8.1\n"
        f"Original error: {str(e)}"
    )

# Now import other dependencies
import os
import streamlit as st
import pinecone
from pinecone import Pinecone
from googletrans import Translator
import google.generativeai as genai
from typing import List, Tuple, Optional, Dict, Any

# Rest of your code continues from here...

# ========== CONFIGURATION ========== #
CONFIG = {
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", ""),
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", ""),
    "INDEX_NAME": "mosdac-rag",
    "EMBEDDING_MODEL": "multi-qa-MiniLM-L6-cos-v1",
    "PINECONE_ENV": "us-east1-gcp",
    "GEMINI_MODEL": "gemini-1.5-flash",
    "TOP_K_RESULTS": 5
}

# ========== INITIALIZATION ========== #
@st.cache_resource(show_spinner=False)
def initialize_services() -> Tuple[Optional[pinecone.Index], Optional[SentenceTransformer]]:
    """Initialize all required services with comprehensive error handling"""
    try:
        # Initialize Gemini
        if not CONFIG["GEMINI_API_KEY"]:
            st.error("Gemini API key not configured")
            return None, None
            
        genai.configure(api_key=CONFIG["GEMINI_API_KEY"])
        
        # Initialize Pinecone
        if not CONFIG["PINECONE_API_KEY"]:
            st.error("Pinecone API key not configured")
            return None, None
            
        pc = Pinecone(api_key=CONFIG["PINECONE_API_KEY"])
        
        # Verify index exists
        try:
            if CONFIG["INDEX_NAME"] not in pc.list_indexes().names():
                available_indexes = pc.list_indexes().names()
                st.error(f"Index '{CONFIG['INDEX_NAME']} not found. Available indexes: {available_indexes}")
                return None, None
        except Exception as e:
            st.error(f"Failed to list Pinecone indexes: {str(e)}")
            return None, None
        
        # Initialize embedding model
        try:
            model = SentenceTransformer(CONFIG["EMBEDDING_MODEL"])
            return pc.Index(CONFIG["INDEX_NAME"]), model
        except Exception as e:
            st.error(f"Failed to initialize embedding model: {str(e)}")
            return None, None
            
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return None, None

# ========== CORE FUNCTIONS ========== #
def query_vector_db(
    query: str, 
    index: pinecone.Index, 
    model: SentenceTransformer, 
    top_k: int = CONFIG["TOP_K_RESULTS"]
) -> List[Dict[str, Any]]:
    """Query Pinecone vector database with enhanced error handling"""
    try:
        query_embedding = model.encode(query, show_progress_bar=False).tolist()
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [
            {
                "source": match["metadata"]["source"],
                "text": match["metadata"]["text"],
                "score": match["score"]
            }
            for match in results["matches"]
        ]
    except Exception as e:
        st.error(f"Vector search failed: {str(e)}")
        return []

def generate_response(question: str, context_data: List[Dict[str, Any]]) -> str:
    """Generate response using Gemini with improved prompt engineering"""
    if not context_data:
        return "⚠ No relevant information found. Please try rephrasing your question."
    
    # Format context with relevance scores
    context = "\n\n".join(
        f"SOURCE: {item['source']}\nRELEVANCE SCORE: {item['score']:.2f}\nCONTENT: {item['text']}"
        for item in context_data
    )
    
    prompt = f"""You are an expert assistant for ISRO's MOSDAC platform. 
Answer the question using ONLY the provided context. Be specific and technical.

Guidelines:
- Use markdown formatting
- Include bullet points for lists
- Reference sources when possible
- If unsure, say "I don't have enough information"

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    try:
        model = genai.GenerativeModel(CONFIG["GEMINI_MODEL"])
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠ Error generating answer: {str(e)}"

def translate_query(query: str) -> str:
    """Handle query translation with fallback"""
    try:
        translator = Translator()
        return translator.translate(query, dest="en").text
    except:
        return query  # Fallback to original query

# ========== STREAMLIT UI ========== #
def setup_ui():
    """Configure Streamlit UI with enhanced styling"""
    st.set_page_config(
        page_title="MOSDAC Knowledge Chatbot",
        page_icon="🛰",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

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
        .context-expander {
            background-color: #1c1e26;
            border: 1px solid #2e3039;
            border-radius: 8px;
            padding: 0.5rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

def display_chat_history():
    """Render chat history with formatted messages"""
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

def display_context(context_data: List[Dict[str, Any]]):
    """Show retrieved context in expander"""
    with st.expander("🔍 Retrieved Context (click to view)", expanded=False):
        for i, item in enumerate(context_data, 1):
            st.markdown(f"""
            <div class="context-expander">
                <strong>Match {i}</strong>  
                <strong>Source:</strong> <code>{item['source']}</code><br>
                <strong>Relevance:</strong> {item['score']:.2f}<br>
                <strong>Content:</strong> {item['text'][:300]}...
            </div>
            """, unsafe_allow_html=True)

# ========== MAIN APPLICATION ========== #
def main():
    setup_ui()
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # UI Components
    st.title("🛰 MOSDAC Knowledge Chatbot")
    st.caption("Ask about ISRO missions, sensors, or documentation")
    
    display_chat_history()
    
    # Process user input
    if user_query := st.chat_input("Type your question here..."):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.rerun()
        
        try:
            with st.spinner("Initializing services..."):
                index, model = initialize_services()
            
            if index and model:
                # Translate query
                translated_query = translate_query(user_query)
                if translated_query != user_query:
                    st.session_state.chat_history.append({
                        "role": "system",
                        "content": f"System: Translated query to English: '{translated_query}'"
                    })
                
                # Vector search
                with st.spinner("Searching knowledge base..."):
                    search_results = query_vector_db(translated_query, index, model)
                    if search_results:
                        display_context(search_results)
                
                # Generate response
                with st.spinner("Generating answer..."):
                    bot_response = generate_response(translated_query, search_results)
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": bot_response
                    })
        
        except Exception as e:
            st.error(f"System error: {str(e)}")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "⚠ Sorry, I encountered an error processing your request."
            })
        
        st.rerun()

if __name__ == "__main__":
    main()