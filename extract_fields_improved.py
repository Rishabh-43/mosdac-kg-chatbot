import pandas as pd
import re

# Load the mission file
#df = pd.read_csv("all_missions.csv")
import logging

logging.basicConfig(level=logging.INFO)
logging.info("Script started")


import os

file_path = "all_missions.csv"
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    print(f"File not found: {file_path}")

# --- Helper functions ---
def extract_launch_date(text):
    # Match dates like "23 Sep 2009", "September 2002", "26 November 2022"
    patterns = [
        r"(launched.*?on\s+)?(\d{1,2}[\s\-]?(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*[\s\-]?\d{4})",
        r"(launched.*?on\s+)?((January|February|March|April|May|June|July|August|September|October|November|December)[\s\-]?\d{4})",
        r"(\d{1,2}[\s\-]?(January|February|March|April|May|June|July|August|September|October|November|December)[\s\-]?\d{4})"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(2).strip()
    return ""

def extract_payloads(text):
    # Match payloads based on known terms
    keywords = ["payload", "instrument", "monitor", "imager", "sounder", "transponder", "altimeter", "scanner"]
    payload_lines = []
    for line in text.split("\n"):
        if any(kw in line.lower() for kw in keywords):
            payload_lines.append(line.strip())
    return " | ".join(payload_lines)

def extract_applications(text):
    # Match blocks related to "used for", "objective", "application", etc.
    keywords = ["applications", "objectives", "used for", "aim", "mission is to", "provides", "intended to"]
    lines = text.split("\n")
    app_lines = []
    for line in lines:
        if any(kw in line.lower() for kw in keywords):
            app_lines.append(line.strip())
    return " | ".join(app_lines)

def extract_orbit_type(text):
    match = re.search(r"(orbit type|orbit)\s*[:\-]?\s*(\w+)", text, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    elif "geostationary" in text.lower():
        return "Geostationary"
    elif "polar" in text.lower():
        return "Polar"
    elif "sun-synchronous" in text.lower():
        return "Sun-synchronous"
    elif "inclined" in text.lower():
        return "Inclined"
    return ""

def extract_status(text):
    if "operational" in text.lower():
        return "Operational"
    elif "decommissioned" in text.lower() or "completed" in text.lower():
        return "Completed"
    elif "commissioned" in text.lower():
        return "Commissioned"
    return "Unknown"

# --- Apply extractions ---
df["launch_date"] = df["description"].apply(extract_launch_date)
df["payloads"] = df["description"].apply(extract_payloads)
df["applications"] = df["description"].apply(extract_applications)
df["orbit_type"] = df["description"].apply(extract_orbit_type)
df["mission_status"] = df["description"].apply(extract_status)

# --- Save result ---
df.to_csv("cleaned_missions.csv", index=False)
print("✅ Cleaned data saved to 'cleaned_missions.csv'")
