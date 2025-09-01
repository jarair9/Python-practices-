import requests
from bs4 import BeautifulSoup

url = "https://bitcointreasuries.net/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Find the table
table = soup.find("table")
rows = table.find_all("tr")

# Extract headers
headers = [th.text.strip() for th in rows[0].find_all("th")]
print("Headers:", headers)

# Extract all data rows
data = []
for row in rows[1:]:  # skip header
    cols = [td.text.strip() for td in row.find_all("td")]
    if cols:
        data.append(cols)

# Print all data nicely
for entry in data:
    print("-" * 60)
    for i in range(len(headers)):
        print(f"{headers[i]}: {entry[i]}")
import csv

# with open("btc_holders.csv", "w", newline='', encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(headers)
#     writer.writerows(data)
