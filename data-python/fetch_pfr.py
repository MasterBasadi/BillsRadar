import requests
from bs4 import BeautifulSoup

standings_url = "https://www.pro-football-reference.com/years/2024/index.htm"
data = requests.get(standings_url)
soup = BeautifulSoup(data.text, features="html.parser")
afc_standings_table = soup.select('table.stats_table')[0]
nfc_standings_table = soup.select('table.stats_table')[1]
print(afc_standings_table)
print(nfc_standings_table)