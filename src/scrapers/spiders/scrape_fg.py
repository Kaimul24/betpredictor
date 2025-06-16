import scrapy
import json
from datetime import date, timedelta
from dotenv import load_dotenv
import os
from utils import daterange
from config import DATES

load_dotenv()
URL = os.getenv("DATA_URL")

class fgSpider(scrapy.Spider):
    name = "fg"

    async def start(self):
        urls = []

        for year in DATES.keys():
            if int(year) == 2021:
                for single_date in daterange(DATES[year][0], DATES[year][1]):
                    formatted_date = single_date.strftime("%Y-%m-%d")
                    url = URL.format(year, year, formatted_date, formatted_date)
                    urls.append(url)

        for url in urls:
            yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        data = json.loads(response.text)
        stats = data.get("data", [])
        
        if stats is None:
            raise KeyError("No stats data found. Check Fangraphs API endpoint/URL")
        
        for row in stats:
            yield row

