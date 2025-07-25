import pandas as pd

# Load main KG edges and mission edges
kg_df = pd.read_csv("final_kg.csv")
mission_edges = pd.read_csv("mission_edges.csv")

# Load metadata tables
geo_df = pd.read_csv("mosdac_coverage_extended.csv")
algo_df = pd.read_csv("mosdac_product_data_with_algorithms.csv")

# Convert geo_df to edge-style rows: (entity → value)
geo_edges = []
for _, row in geo_df.iterrows():
    subject = f"{row['mission_name']}:{row['sensor']}"
    for attr in ["min_lat", "max_lat", "min_lon", "max_lon", "resolution"]:
        if pd.notna(row[attr]):
            geo_edges.append({"source": subject, "relation": attr, "target": str(row[attr])})

geo_edges_df = pd.DataFrame(geo_edges)

# Convert algo_df to edge-style rows
algo_edges = []
for _, row in algo_df.iterrows():
    subject = row["product_name"]
    for col in algo_df.columns:
        if col != "product_name" and pd.notna(row[col]):
            algo_edges.append({"source": subject, "relation": col, "target": str(row[col])})

algo_edges_df = pd.DataFrame(algo_edges)

# Combine everything
kg_master = pd.concat([kg_df, mission_edges, geo_edges_df, algo_edges_df], ignore_index=True)
kg_master.drop_duplicates(inplace=True)
kg_master.reset_index(drop=True, inplace=True)

# Save final KG
kg_master.to_csv("kg_master.csv", index=False)
print("✅ Final KG saved as kg_master.csv")
