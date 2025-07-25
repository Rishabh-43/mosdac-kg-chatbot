import pandas as pd

# Load both files
kg = pd.read_csv("final_kg.csv")
geo = pd.read_csv("mosdac_coverage_extended.csv")

print("📄 Columns in final_kg.csv:")
print(kg.columns.tolist())

print("\n📄 Columns in mosdac_coverage_extended.csv:")
print(geo.columns.tolist())
