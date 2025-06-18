import scrapy
import json
from dotenv import load_dotenv
import os
from utils import daterange
from config import DATES
from scrapers.items import BatterStat, PitcherStat
from datetime import datetime

load_dotenv()
BATTER_URL = os.getenv("BATTER_URL")
PITCHER_URL = os.getenv("PITCHER_URL")

class fgSpider(scrapy.Spider):
    name = "fg"

    async def start(self):
        self.requested_dates = set()
        yield scrapy.Request(
            "https://www.fangraphs.com/",
            meta={"playwright": True,
                  "playwright_include_page": True,
                  "playwright_page_goto_kwargs": {
                    "wait_until": "domcontentloaded",
                    "timeout": 45_000,
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
            "Referer": "https://www.fangraphs.com/leaders/",
            "Origin":  "https://www.fangraphs.com",
            "Sec-Fetch-Mode": "cors",
        }

        for year, (d0, d1) in DATES.items():
            for d in daterange(d0, d1):
                day = d.strftime("%Y-%m-%d")
                self.requested_dates.add(day)
                url_batter = BATTER_URL.format(year, year, day, day)
                url_pitcher = PITCHER_URL.format(year, year, day, day)

                yield scrapy.Request(
                    url_batter,
                    cookies=self.cookies,
                    headers=json_headers,
                    callback=self.parse,
                    cb_kwargs={"date": day},
                )

                yield scrapy.Request(
                    url_pitcher,
                    cookies=self.cookies,
                    headers=json_headers,
                    callback=self.parse,
                    cb_kwargs={"date": day},
                )


    def parse(self, response, date):
        if response.status == 403:
            self.logger.error(f"Request failed with status 403 (Forbidden) for date: {date}.")
            return
        
        payload = json.loads(response.text)

        if payload['totalCount'] == 0:
            self.logger.warning(f"No stats data found for date: {date}")
            return

        stats = payload.get("data", [])

        for row in stats:
            keys = list(row.keys())
            type = keys[0]

            if type == 'Bats':
                item = BatterStat()
                item['player_id'] = row['playerid']
                item['name'] = row['PlayerName']
                item['team'] = row['TeamName']
                item['games'] = row['G']
                item['ab'] = row['AB']
                item['pa'] = row['PA']
                item['ops'] = row['OPS']
                item['bb_k'] = row['BB/K']
                item['wrc_plus'] = row['wRC+']
                item['barrel_percent'] = row['Barrel%']
                item['hard_hit'] = row['HardHit%']
                item['war'] = row['WAR']
                item['baserunning'] = row['BaseRunning']
                item['date'] = date
                item['scraped_at'] = datetime.now()
                yield item
            
            elif type == 'Throws':
                item = PitcherStat()
                item['player_id'] = row['playerid']
                item['name'] = row['PlayerName']
                item['team'] = row['TeamName']
                item['games'] = row['G']
                item['era'] = row['ERA']
                item['ip'] = row['IP']
                item['k_percent'] = row['K%']
                item['bb_percent'] = row['BB%']
                item['barrel_percent'] = row['Barrel%']
                item['hard_hit'] = row['HardHit%']
                item['war'] = row['WAR']
                item['siera'] = row['SIERA']
                item['fip'] = row['FIP']
                item['date'] = date
                item['scraped_at'] = datetime.now()
                yield item
            

