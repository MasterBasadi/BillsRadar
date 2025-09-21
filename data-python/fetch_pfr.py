import requests
from bs4 import BeautifulSoup
import pandas as pd

standings_url = "https://www.pro-football-reference.com/years/2024/index.htm"
data = requests.get(standings_url)
soup = BeautifulSoup(data.text, features="html.parser")
afc_standings_table = soup.select('table.stats_table')[0]
nfc_standings_table = soup.select('table.stats_table')[1]

afc_links = afc_standings_table.find_all('a')
nfc_links = nfc_standings_table.find_all('a')

afc_links = [al.get("href") for al in afc_links]
afc_links = [al for al in afc_links if '/teams/' in al]

nfc_links = [nl.get("href") for nl in nfc_links]
nfc_links = [nl for nl in nfc_links if '/teams/' in nl]

links = afc_links + nfc_links
teams_urls = [f"https://www.pro-football-reference.com{l}" for l in links]

team_url = teams_urls[0]
data = requests.get(team_url)
matches = pd.read_html(data.text, match="Schedule & Game Results")

print(matches[0])