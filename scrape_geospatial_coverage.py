import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# List of product discovery URLs and related sensors
products = [
    {"mission": "OCEANSAT-3", "sensor": "OCM-3", "url": "https://mosdac.gov.in/discovery/L2OCM_LAC_AD.jsp"},
    {"mission": "OCEANSAT-3", "sensor": "SCAT-3", "url": "https://mosdac.gov.in/discovery/L2SCAT_LAC_AD.jsp"},
    {"mission": "OCEANSAT-3", "sensor": "SSTM", "url": "https://mosdac.gov.in/discovery/L2SSTM_LAC_AD.jsp"},
    {"mission": "INSAT-3D", "sensor": "Imager & Sounder", "url": "https://mosdac.gov.in/discovery/INSAT3D_L2.jsp"},
    {"mission": "INSAT-3DR", "sensor": "Imager & Sounder", "url": "https://mosdac.gov.in/discovery/INSAT3DR_L2.jsp"},
]

# Set up headless browser
options = Options()
options.headless = True
driver = webdriver.Chrome(options=options)

# Output CSV
csv_file = open("mosdac_coverage.csv", mode="w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow(["mission_name", "sensor", "min_lat", "max_lat", "min_lon", "max_lon", "resolution"])

for item in products:
    print(f"🔍 Scanning {item['url']} …")
    driver.get(item["url"])
    time.sleep(5)  # Let the page load

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Attempt to extract metadata from the text (varies by page)
    text = soup.get_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    min_lat = max_lat = min_lon = max_lon = resolution = ""

    for line in lines:
        if "coverage" in line.lower() or "area" in line.lower():
            if any(c in line for c in ['°', 'deg', 'latitude', 'longitude']):
                print(f"  → Found candidate line: {line}")
        if "resolution" in line.lower():
            resolution = line

    writer.writerow([item["mission"], item["sensor"], min_lat, max_lat, min_lon, max_lon, resolution])

driver.quit()
csv_file.close()
print("✅ Saved to mosdac_coverage.csv")
