import scrapy
import json
from config import DATES
import os

from dotenv import load_dotenv

load_dotenv()
LG_AVG_URL = os.getenv("LG_AVG_URL")

class LeagueAverageSpider(scrapy.Spider):
    name = 'league_avg'

    async def start(self):
        self.year_data = {}

        self.json_headers = {
            **self.settings.getdict("DEFAULT_REQUEST_HEADERS"),
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": "https://www.fangraphs.com",
            "Origin":  "https://www.fangraphs.com",
            "Sec-Fetch-Mode": "cors",
        }

        yield scrapy.Request(
            "https://www.fangraphs.com/",
            meta={
                "playwright": True,
                "playwright_include_page": True,
                "playwright_page_goto_kwargs": {
                    "wait_until": "domcontentloaded",
                    "timeout": 45_000,
                },
            },
            callback=self.after_handshake,
            dont_filter=True,
        )

    async def after_handshake(self, response):
        page = response.meta["playwright_page"]
        cookies = await page.context.cookies()
        self.cookies = {c["name"]: c["value"] for c in cookies}
        await page.close()

        for year in DATES.keys():
            batter_url = LG_AVG_URL.format('bat', year, year)
            pitcher_url = LG_AVG_URL.format('pit', year, year)

            yield scrapy.Request(
                        batter_url,
                        cookies=self.cookies,
                        headers=self.json_headers,
                        callback=self.parse,
                        cb_kwargs={"year": year, "type": 'Bats'},
                    )
            
            yield scrapy.Request(
                        pitcher_url,
                        cookies=self.cookies,
                        headers=self.json_headers,
                        callback=self.parse,
                        cb_kwargs={"year": year, "type": 'Throws'},
                    )
            
    def parse(self, response, year, type):
        payload = json.loads(response.text)

        if payload['totalCount'] == 0:
            self.logger.warning(f"No stats data found for: {year}")
            return

        data = payload.get("data", [])
        stats = data[0]

        # Initialize year data if not exists
        if year not in self.year_data:
            self.year_data[year] = {'year': year}

        if type == 'Bats':
            self.year_data[year]['Bats'] = {
                'type': type,
                'ops': stats['OPS'],
                'babip': stats['BABIP'],
                'bb_k': stats['BB/K'],
                'woba': stats['wOBA'],
                'barrel_percent': stats['Barrel%'],
                'hard_hit': stats['HardHit%']
            }
        
        if type == 'Throws':
            self.year_data[year]['Throws'] = {
                'type': type,
                'era': stats['ERA'],
                'babip': stats['BABIP'],
                'k_percent': stats['K%'],
                'bb_percent': stats['BB%'],
                'barrel_percent': stats['Barrel%'],
                'hard_hit': stats['HardHit%'],
                'siera': stats['SIERA'],
                'fip': stats['FIP']
            }

        if 'Bats' in self.year_data[year] and 'Throws' in self.year_data[year]:
            yield self.year_data[year]
            


        
 
            

