# scrape_products_sensors.py

import csv, time
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By

# ——— Configure EdgeDriver ———
edge_driver_path = "C:/Users/risha/OneDrive/Desktop/project/edgedriver_win64 (2)/msedgedriver.exe"
service = Service(edge_driver_path)
options = webdriver.EdgeOptions()
options.add_argument("start-maximized")
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
)

driver = webdriver.Edge(service=service, options=options)

# ——— Open the MOSDAC Discovery/Product page ———
driver.get("https://mosdac.gov.in/discovery/discovery.jsp")
time.sleep(5)  # wait for the DataTable to initialize

# ——— Locate the main products table rows ———
# This selector grabs all <tr> in the table body
rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")

print(f"Found {len(rows)} product rows.")

# ——— Write CSV ———
with open("mosdac_product_data_with_sensors.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["product_name", "url", "sensor_info"])

    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")
        if len(cells) < 2:
            continue

        # 1️⃣ Product name & URL in first cell
        link = cells[0].find_element(By.TAG_NAME, "a")
        product_name = link.text.strip()
        product_url = link.get_attribute("href")

        # 2️⃣ Sensor info often in the 3rd or 4th column—adjust index if needed
        sensor_info = cells[2].text.strip() if len(cells) > 2 else ""

        writer.writerow([product_name, product_url, sensor_info])

driver.quit()
print("✅ mosdac_product_data_with_sensors.csv created with product entries.")
