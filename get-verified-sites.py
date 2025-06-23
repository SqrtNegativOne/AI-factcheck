import requests
from bs4 import BeautifulSoup
import pickle

URL = "https://en.wikipedia.org/wiki/Wikipedia:Reliable_sources/Perennial_sources"
response = requests.get(URL)
soup = BeautifulSoup(response.text, "html.parser")

urls = []

for table in soup.select(".mw-parser-output table"):
    for row in table.select("tr"):

        cells = row.find_all("td")

        if len(cells) != 6:
            continue
            
        if str(cells[1]).find("Generally reliable") == -1:
            continue
        
        
        if not cells[5].a:
            continue

        
        for span in cells[5].find_all("span"):
            for a in span.find_all("a"):
                if a.get("title", "") == "HTTPS links":
                    urls.append(a.get("href", "")[25:])


print(urls)

with open("reliable_sources.txt", "w", encoding="utf-8") as f:
    for url in urls:
        f.write(url + "\n")

with open("reliable_sources.pkl", "wb") as f:
    pickle.dump(urls, f)