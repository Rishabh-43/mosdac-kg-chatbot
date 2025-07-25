import os
import pandas as pd

# Folder containing all mission CSVs
input_folder = "missions_csv"
output_file = "all_missions.csv"

# List to store all DataFrames
all_missions = []

# Loop through each CSV in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)
        all_missions.append(df)
        print(f"✅ Loaded: {filename}")

# Merge all into one DataFrame
merged_df = pd.concat(all_missions, ignore_index=True)

# Save to final merged CSV
merged_df.to_csv(output_file, index=False)
print(f"\n✅ Merged all files into: {output_file}")
