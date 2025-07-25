# extract_algorithms.py

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.service import Service
import time

df = pd.read_csv("mosdac_product_data_with_sensors.csv")

svc = Service("C:/Users/risha/OneDrive/Desktop/project/edgedriver_win64 (2)/msedgedriver.exe")
opts = webdriver.EdgeOptions()
opts.add_argument("headless")
driver = webdriver.Edge(service=svc, options=opts)

algorithms = []
formats = []

for idx, row in df.iterrows():
    url = row["product_url"]
    driver.get(url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    alg = soup.select_one(".product-algo").text.strip() if soup.select_one(".product-algo") else ""
    fmt = soup.select_one(".product-format").text.strip() if soup.select_one(".product-format") else ""
    
    algorithms.append(alg)
    formats.append(fmt)

driver.quit()

df["processing_algorithm"] = algorithms
df["data_format"] = formats
df.to_csv("mosdac_product_data_with_algorithms.csv", index=False)
print("✅ Saved mosdac_product_data_with_algorithms.csv")
