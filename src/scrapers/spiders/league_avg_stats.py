import scrapy
from config import DATES
import os

from dotenv import load_dotenv
from src.scrapers.fangraphs_session import (
    FANGRAPHS_SPIDER_SETTINGS,
    fangraphs_headers,
    fangraphs_request,
    load_fangraphs_json,
    require_fangraphs_storage_state,
)

load_dotenv()
LG_AVG_URL = os.getenv("LG_AVG_URL")

class LeagueAverageSpider(scrapy.Spider):
    name = 'league_avg'
    custom_settings = FANGRAPHS_SPIDER_SETTINGS.copy()

    async def start(self):
        require_fangraphs_storage_state(self)
        self.year_data = {}
        self.json_headers = fangraphs_headers(self.settings)

        for year in DATES.keys():
            batter_url = LG_AVG_URL.format('bat', year, year)
            pitcher_url = LG_AVG_URL.format('pit', year, year)

            yield fangraphs_request(
                        batter_url,
                        headers=self.json_headers,
                        callback=self.parse,
                        cb_kwargs={"year": year, "type": 'Bats'},
                    )
            
            yield fangraphs_request(
                        pitcher_url,
                        headers=self.json_headers,
                        callback=self.parse,
                        cb_kwargs={"year": year, "type": 'Throws'},
                    )
            
    def parse(self, response, year, type):
        payload = load_fangraphs_json(response, self.logger)

        if payload['totalCount'] == 0:
            self.logger.warning(f"No stats data found for: {year}")
            return

        data = payload.get("data", [])
        stats = data[0]

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
                'hard_hit': stats['HardHit%'],
                'ev': stats['EV'],
                'iso': stats['ISO'],
                'gb_fb': stats['GB/FB']
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
                'fip': stats['FIP'],
                'ev': stats['EV'],
                'hr_fb': stats['HR/FB'],
                'gmli': stats['gmLI']
            }

        if 'Bats' in self.year_data[year] and 'Throws' in self.year_data[year]:
            yield self.year_data[year]
            


        
 
            
