import pandas as pd
from pathlib import Path

# Config - Point to current directory (no 'data/processed' needed)
INPUT_DIR = Path(__file__).parent  # Looks in the same folder as the script

def load_and_standardize_edges():
    """Load files with THEIR ACTUAL NAMES"""
    print("\nLoading files:", list(INPUT_DIR.glob("*.csv")))
    
    # Load files with their exact names
    mission_edges = pd.read_csv(INPUT_DIR / "cleaned_missions.csv")
    coverage_edges = pd.read_csv(INPUT_DIR / "mosdac_coverage_extended.csv")
    product_algo = pd.read_csv(INPUT_DIR / "mosdac_product_data_with_algorithms.csv")
    
    # Standardize columns (adjust as needed)
    product_algo_edges = product_algo.rename(columns={
        "product": "source",
        "algorithm": "target"
    }).assign(relation="uses_algorithm")
    
    return pd.concat([mission_edges, coverage_edges, product_algo_edges], ignore_index=True)

def main():
    kg_edges = load_and_standardize_edges()
    
    # Create output folder if it doesn't exist
    OUTPUT_DIR = Path("output")  # New folder for results
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    kg_edges.to_csv(OUTPUT_DIR / "final_kg_edges.csv", index=False)
    print(f"\n✅ Merged KG saved to: {OUTPUT_DIR / 'final_kg_edges.csv'}")

if __name__ == "__main__":
    main()