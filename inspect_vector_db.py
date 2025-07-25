import pickle
import faiss

# Load metadata
with open("vector_db/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print("🧠 Sample metadata entry:")
print(metadata[0])  # Print first item

# Load FAISS index
index = faiss.read_index("vector_db/faiss_index.bin")
print(f"\n📊 Total vectors in index: {index.ntotal}")
