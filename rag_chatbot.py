# -*- coding: utf-8 -*-
import sys

try:
    import torch
    from sentence_transformers import SentenceTransformer
    import scipy
    import nltk
except ImportError as e:
    raise ImportError(
        "\nMissing dependencies. Please run:\n"
        "pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html\n"
        "pip install sentence-transformers==2.2.2\n"
        "pip install scipy==1.10.1\n"
        "pip install nltk==3.8.1\n"
        f"\nOriginal error: {str(e)}"
    ) from None

import os
import streamlit as st
from pinecone import Pinecone
from googletrans import Translator
import google.generativeai as genai
from typing import List, Tuple, Optional, Dict, Any

CONFIG = {
    "PINECONE_API_KEY": st.secrets.get("PINECONE_API_KEY", "pcsk_7USieR_TWCp5dcfXBN4ePuRgXGuYCR2YDX8ipq2V43vc4My6mZZUn8VffNFoUgeN2ZYNL2"),
    "GEMINI_API_KEY": st.secrets.get("GEMINI_API_KEY", "AIzaSyAyhcpDKXBCs9J5B-lS_-RCbrCNcfKP7hw"),
    "INDEX_NAME": "mosdac-rag",
    "EMBEDDING_MODEL": "multi-qa-MiniLM-L6-cos-v1",
    "PINECONE_ENV": "us-east1-gcp",
    "GEMINI_MODEL": "gemini-1.5-flash",
    "TOP_K_RESULTS": 5
}

@st.cache_resource(show_spinner=False)
def initialize_services() -> Tuple[Optional[object], Optional[SentenceTransformer]]:
    try:
        genai.configure(api_key=CONFIG["GEMINI_API_KEY"])
        pc = Pinecone(api_key=CONFIG["PINECONE_API_KEY"])
        if CONFIG["INDEX_NAME"] not in pc.list_indexes().names():
            return None, None
        index = pc.Index(CONFIG["INDEX_NAME"])
        model = SentenceTransformer(CONFIG["EMBEDDING_MODEL"])
        return index, model
    except Exception as e:
        st.error(f"Service initialization failed: {str(e)}")
        return None, None

def query_vector_db(query: str, index: object, model: SentenceTransformer, top_k: int = CONFIG["TOP_K_RESULTS"]) -> List[Dict[str, Any]]:
    try:
        query_vector = model.encode(query, show_progress_bar=False).tolist()
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
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
    if not context_data:
        return "⚠ No relevant information found. Please try rephrasing your question."

    context = "\n\n".join(
        f"SOURCE: {item['source']}\nRELEVANCE SCORE: {item['score']:.2f}\nCONTENT: {item['text']}"
        for item in context_data
    )

    prompt = f"""
You are an expert assistant for ISRO's MOSDAC platform. 
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:"""

    try:
        model = genai.GenerativeModel(CONFIG["GEMINI_MODEL"])
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠ Error generating answer: {str(e)}"

def translate_query(query: str) -> str:
    try:
        translator = Translator()
        return translator.translate(query, dest="en").text
    except:
        return query

def setup_ui():
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
    for message in st.session_state.chat_history:
        css_class = "message-user" if message["role"] == "user" else "message-bot"
        icon = "👤 You:" if message["role"] == "user" else "🤖 Assistant:"
        st.markdown(f"""
        <div class="{css_class}">
            <strong>{icon}</strong> {message["content"]}
        </div>
        """, unsafe_allow_html=True)

def display_context(context_data: List[Dict[str, Any]]):
    with st.expander("🔍 Retrieved Context", expanded=False):
        for i, item in enumerate(context_data, 1):
            st.markdown(f"""
            <div class="context-expander">
                <strong>Match {i}</strong><br>
                <strong>Source:</strong> <code>{item['source']}</code><br>
                <strong>Relevance:</strong> {item['score']:.2f}<br>
                <strong>Content:</strong> {item['text'][:300]}...
            </div>
            """, unsafe_allow_html=True)

def main():
    setup_ui()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("🛰 MOSDAC Knowledge Chatbot")
    st.caption("Ask about ISRO missions, sensors, or documentation")

    display_chat_history()

    if user_query := st.chat_input("Type your question here..."):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.rerun()

        with st.spinner("Loading..."):
            index, model = initialize_services()

        if index and model:
            translated_query = translate_query(user_query)
            if translated_query != user_query:
                st.session_state.chat_history.append({
                    "role": "system",
                    "content": f"System: Translated to English → '{translated_query}'"
                })

            with st.spinner("Searching..."):
                search_results = query_vector_db(translated_query, index, model)
                if search_results:
                    display_context(search_results)

            with st.spinner("Answering..."):
                response = generate_response(translated_query, search_results)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })

        st.rerun()

if __name__ == "__main__":
    main()
