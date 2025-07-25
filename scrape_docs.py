from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
import csv
import time

# ——— Your EdgeDriver configuration ———
edge_driver_path = "C:/Users/risha/OneDrive/Desktop/project/edgedriver_win64 (2)/msedgedriver.exe"
service = Service(edge_driver_path)
options = webdriver.EdgeOptions()
options.add_argument("start-maximized")
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
)

driver = webdriver.Edge(service=service, options=options)

# ——— Pages to scan for PDFs ———
pages = [
    "https://mosdac.gov.in/announcements",
    "https://mosdac.gov.in/insitu",
    "https://mosdac.gov.in/atlases",
    "https://mosdac.gov.in/sitemap"
]

# ——— Open CSV for writing ———
with open("mosdac_docs.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["page", "title", "url"])

    for page in pages:
        print(f"🔍 Scanning {page} …")
        driver.get(page)
        time.sleep(5)  # allow JS to load

        # find all <a> tags ending with .pdf
        links = driver.find_elements(By.CSS_SELECTOR, "a[href$='.pdf']")
        print(f"  → Found {len(links)} PDFs")

        for a in links:
            href = a.get_attribute("href")
            title = a.text.strip() or href.split("/")[-1]
            writer.writerow([page, title, href])

driver.quit()
print("✅ mosdac_docs.csv created successfully.")
