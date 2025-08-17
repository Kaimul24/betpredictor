import csv, os, scrapy
from io import StringIO
from scrapers.items import FRVItem
from dotenv import load_dotenv
from config import DATES, INT_TO_MONTH_MAP_FIELDING
from src.utils import normalize_names

load_dotenv()
FRV_URL = os.getenv("FRV_URL")

class fieldingSpider(scrapy.Spider):
    name = "fielding"

    async def start(self):
        self.json_headers = {
            **self.settings.getdict("DEFAULT_REQUEST_HEADERS"),
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": "https://www.baseballsavant/leaderboard/fielding_run_value",
            "Origin":  "https://www.baseballsavant.com",
            "Sec-Fetch-Mode": "cors",
        }

        yield scrapy.Request(
            "https://www.baseballsavant.com",
            meta={
                "playwright": True,
                "playwright_include_page": True,
                "playwright_page_goto_kwargs": {
                    "wait_until": "domcontentloaded",
                    "timeout": 45_000,
                },
            },
            callback=self.after_handshake
        )
    
    async def after_handshake(self, response):
        page = response.meta["playwright_page"]
        cookies = await page.context.cookies()
        self.cookies = {c["name"]: c["value"] for c in cookies}
        await page.close()

        for year in DATES.keys():
            url = FRV_URL.format(year, year)
            yield scrapy.Request(
                    url,
                    cookies=self.cookies,
                    headers=self.json_headers,
                    callback=self.parse_leaderboard,
                    cb_kwargs={"year": year},
                )

    def parse_leaderboard(self, response, year):
        csv_file = StringIO(response.text)
        csv_reader = csv.DictReader(csv_file)
        data = [row for row in csv_reader]

        if data == []:
            self.logger.warning(f"No fielding data found for {year}")
            return
        
        for row in data:
            item = FRVItem()
            raw_name = row['name']
            name = ' '.join(reversed(raw_name.split(', ')))
            
            item['name'] = name
            item['normalized_player_name'] = normalize_names(name)
            item['season'] = year
            item['month'] = row['api_game_date_month_mm']
            item['frv'] = round(float(row['total_runs']), 2)

            item['total_innings'] = int(row['outs_total']) / 3
            item['innings_C'] = int(row['outs_2']) / 3
            item['innings_1B'] = int(row['outs_3']) / 3
            item['innings_2B'] = int(row['outs_4']) / 3
            item['innings_3B'] = int(row['outs_5']) / 3
            item['innings_SS'] = int(row['outs_6']) / 3
            item['innings_LF'] = int(row['outs_7']) / 3
            item['innings_CF'] = int(row['outs_8']) / 3
            item['innings_RF'] = int(row['outs_9']) / 3
            yield item

    
