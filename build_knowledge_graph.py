import pandas as pd
import networkx as nx

# Load structured mission data
df = pd.read_csv("cleaned_missions.csv")

# Create a directed graph
G = nx.DiGraph()

# Loop through rows and build triples
for _, row in df.iterrows():
    mission = row["mission_name"]

    # Add basic attributes
    if pd.notna(row["launch_date"]):
        G.add_edge(mission, row["launch_date"], relation="launchedOn")
    if pd.notna(row["orbit_type"]):
        G.add_edge(mission, row["orbit_type"], relation="hasOrbitType")
    if pd.notna(row["mission_status"]):
        G.add_edge(mission, row["mission_status"], relation="hasStatus")
    
    # Payloads can be multiple
    if pd.notna(row["payloads"]):
        for p in row["payloads"].split(","):
            payload = p.strip()
            if payload:
                G.add_edge(mission, payload, relation="hasPayload")

    # Applications can also be multiple
    if pd.notna(row["applications"]):
        for a in row["applications"].split(","):
            app = a.strip()
            if app:
                G.add_edge(mission, app, relation="usedFor")

print(f"✅ Knowledge graph built with {len(G.nodes)} nodes and {len(G.edges)} edges.")

# Optional: Save as edge list CSV
nx.write_edgelist(G, "mission_kg_edges.csv", delimiter=",", data=["relation"])
print("📁 Edge list saved as: mission_kg_edges.csv")
