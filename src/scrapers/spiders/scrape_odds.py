from config import DATES
from dotenv import load_dotenv
import json, os, scrapy
from itertools import islice
from utils import daterange
from scrapers.items import OddsItem

load_dotenv()
ODDS_URL = os.getenv("ODDS_URL")

class oddsSpider(scrapy.Spider):
    name = 'odds'

    async def start(self):
        self.requested_dates = set()

        yield scrapy.Request(
            "https://www.sportsbookreview.com/betting-odds/",
            meta={"playwright": True,
                  "playwright_include_page": True,
                  "playwright_page_goto_kwargs": {
                    "wait_until": "domcontentloaded",
                    "timeout": 10_000,
                },
            },
            headers={}, 
            callback=self.after_handshake,
            dont_filter=True,
        )
    
    async def after_handshake(self, response):
        page = response.meta["playwright_page"]
        cookies = await page.context.cookies()
        self.cookies = {c["name"]: c["value"] for c in cookies}
        await page.close()

        json_headers = {
            **self.settings.getdict("DEFAULT_REQUEST_HEADERS"),
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": "https://www.sportsbookreview.com/betting-odds/mlb-baseball/",
            "Origin":  "https://www.sportsbookreview.com/betting-odds/",
            "Sec-Fetch-Mode": "cors",
        }

        for year, (d0, d1) in DATES.items():
            for d in daterange(d0, d1):
                day = d.strftime("%Y-%m-%d")
                self.requested_dates.add(day)
                url = ODDS_URL.format(day)

                yield scrapy.Request(
                    url,
                    cookies=self.cookies,
                    headers=json_headers,
                    callback=self.parse,
                    cb_kwargs={"date": day, "year": year},
                )

    def parse(self, response, date, year):
        payload = json.loads(response.text)
        x = payload['pageProps']['oddsTables']

        if x == []:
            self.logger.warning(f"No odds data found for date: {date}")
            return
        
        y = x[0]
        odds_table = y['oddsTableModel']
        games = odds_table['gameRows']

        for game in games:
            game_data = game['gameView']

            away_team = game_data['awayTeam']['shortName']
            home_team = game_data['homeTeam']['shortName']

            if away_team == 'AL' or home_team == 'AL':
                self.logger.warning(f"Skipping odds for All Star Game on: {date}")
                return
            
            away_starter_dict = game_data['awayStarter']
            if away_starter_dict == None and game_data['gameId'] == 354113:
                away_starter = 'Michael Lorenzen'
            else:
                away_starter = ' '.join(islice(away_starter_dict.values(), 2))

            home_starter_dict = game_data['homeStarter']
            if home_starter_dict == None and game_data['gameId'] == 354113:
                home_starter = 'JP Sears'
            else:
                home_starter = ' '.join(islice(home_starter_dict.values(), 2))

            away_score = game_data['awayTeamScore']
            home_score = game_data['homeTeamScore']

            winner = away_team if away_score > home_score else home_team

            game_odds = game['oddsViews']

            for odds in game_odds:
                if odds is not None:
                    item = OddsItem()
                    item['date'] = date
                    item['away_team'] = away_team
                    item['home_team'] = home_team
                    item['away_starter'] = away_starter
                    item['home_starter'] = home_starter
                    item['away_score'] = away_score
                    item['home_score'] = home_score
                    item['winner'] = winner
                    item['sportsbook'] = odds['sportsbook']
                    item['away_odds'] = odds['openingLine']['awayOdds']
                    item['home_odds'] = odds['openingLine']['homeOdds']
                    item['season'] = year
                    yield item