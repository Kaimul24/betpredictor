from config import DATES
from scrapers.items import ParkFactorItem

from dotenv import load_dotenv
from datetime import datetime
import json, os, scrapy, re

load_dotenv()
PF_URL = os.getenv("PF_URL")

class pfSpider(scrapy.Spider):
    name = 'park_factor'

    async def start(self):
        yield scrapy.Request(
            "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?",
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
            "Referer": "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?",
            "Origin":  "https://baseballsavant.mlb.com",
            "Sec-Fetch-Mode": "cors",
        }

        for year in DATES.keys():
            url = PF_URL.format(year)

            yield scrapy.Request(
                url,
                cookies=self.cookies,
                headers=json_headers,
                callback=self.parse,
                cb_kwargs={'year': year}
            )

    def parse(self, response, year):
        script_text = "".join(response.css("div.article-template script::text").getall())

        if not script_text:
            script_text = response.text

        data = re.search(r"var\s+data\s*=\s*(\[[\s\S]*?\])\s*;", script_text)
        if not data:
            self.logger.error("Couldn't find `var data = [...]` on the page.")
            return

        raw_json = data.group(1)
        rows = json.loads(raw_json)

        for row in rows:
            item = ParkFactorItem()
            item['venue_id'] = row['venue_id']
            item['venue_name'] = row['venue_name']
            item['season'] = year
            item['park_factor'] = row['index_woba']
            item['scraped_at'] = datetime.now()
            yield item
                


