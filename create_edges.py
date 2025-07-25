# create_edges.py (WORKING VERSION)
import pandas as pd
from ast import literal_eval

print("🔍 Loading cleaned_missions.csv...")
missions = pd.read_csv("cleaned_missions.csv")

# Convert stringified lists to actual lists (if needed)
try:
    missions['payloads'] = missions['payloads'].apply(literal_eval)
except:
    pass  # Skip if already in list format

# Create mission-payload edges
mission_edges = missions.explode('payloads').rename(columns={
    'mission_name': 'source',
    'payloads': 'target'
})[['source', 'target']].dropna()
mission_edges['relation'] = 'has_payload'

print(f"\n🌟 First 5 relationships:\n{mission_edges.head(5)}")
mission_edges.to_csv("mission_edges.csv", index=False)
print("\n✅ Saved mission_edges.csv")