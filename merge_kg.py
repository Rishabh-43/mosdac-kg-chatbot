# merge_kg.py
import pandas as pd

# Load mission edges (the working file)
mission_edges = pd.read_csv("mission_edges.csv")

# Save as final KG (we'll add more later)
mission_edges.to_csv("final_kg.csv", index=False)
print("✅ Saved final_kg.csv with mission-payload relationships!")