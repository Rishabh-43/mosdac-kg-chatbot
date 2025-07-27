import pinecone
from sentence_transformers import SentenceTransformer

def init_pinecone(api_key: str, environment: str = "us-east1-gcp"):
    """Initialize Pinecone client"""
    try:
        pc = pinecone.Pinecone(api_key=api_key)
        return pc
    except Exception as e:
        raise ConnectionError(f"Failed to initialize Pinecone: {str(e)}")

def get_embeddings_model(model_name: str = "multi-qa-MiniLM-L6-cos-v1"):
    """Load sentence transformer model"""
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        raise ImportError(f"Failed to load model: {str(e)}")

def query_index(query: str, 
               index_name: str = "mosdac-rag", 
               top_k: int = 3,
               api_key: str = None):
    """Query Pinecone index with text"""
    try:
        pc = init_pinecone(api_key)
        index = pc.Index(index_name)
        model = get_embeddings_model()
        query_embedding = model.encode(query).tolist()
        return index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )['matches']
    except Exception as e:
        raise RuntimeError(f"Query failed: {str(e)}")