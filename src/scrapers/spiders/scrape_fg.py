import scrapy
import json
from dotenv import load_dotenv
import os
from config import DATES, LG_AVG_STATS
from scrapers.items import BatterStat, PitcherStat
from datetime import datetime
import numpy as np

load_dotenv()
TEAM_URL = os.getenv("TEAM_URL")
GAME_LOG_URL = os.getenv("GAME_URL")

class fgSpider(scrapy.Spider):
    name = "fg"
    
    async def start(self):
        self.requested_dates = set()
        self.LG_AVG_STATS = LG_AVG_STATS

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

        self.seen_ids = set()

        for team_id in range(1, 31):
            for year in DATES.keys():
                url = TEAM_URL.format(team_id, year)
                yield scrapy.Request(
                    url,
                    cookies=self.cookies,
                    headers=self.json_headers,
                    callback=self.parse_roster,
                    cb_kwargs={"year": year},
                )

    def parse_roster(self, response, year):
        payload = json.loads(response.text)

        if payload == []:
            self.logger.warning(f"No roster data found for {year}")
            return

        for g in payload:
            lineup = g['lineupInfo']

            for p in lineup:
                id = p['playerId']
                pos_str = p['position']
                pos_list = pos_str.split("-")
                # May not need this stuff
                if "CF" in pos_list or "RF" in pos_list or "LF" in pos_list:
                    pos = "OF"
                else:
                    pos = pos_list[0]

                if id in self.seen_ids:
                    continue

                self.seen_ids.add(id)
                url = GAME_LOG_URL.format(id, pos, year)

                yield scrapy.Request(
                    url,
                    cookies=self.cookies,
                    headers=self.json_headers,
                    cb_kwargs={"year": year, "id": id},
                    callback=self.parse_game_log,
                )

    def parse_game_log(self, response, year, id):
        
        payload = json.loads(response.text)

        stats = payload.get("mlb", [])

        if stats == []:
            self.logger.warning(f"No stats data found for player ID: {id} in {year}")
            return

        for game in stats[1:]:
            keys = list(game.keys())
            type = keys[7]
            events = game['Events']

            if events > 0:
                if type == 'G':
                    
                    item = BatterStat()
                    item['player_id'] = game['playerid']
                    item['name'] = game['PlayerName']
                    item['team'] = game['Team']
                    item['dh'] = game['dh']
                    item['ab'] = game['AB']
                    item['pa'] = game['PA']

                    ops = game.get("OPS%", LG_AVG_STATS[str(year)]['Bats']['ops'])
                    item['ops'] = ops

                    bb_k = game.get("BB/K%", LG_AVG_STATS[str(year)]['Bats']['bb_k'])
                    item['bb_k'] = bb_k

                    wrc_plus = game.get("wRC+", 100)
                    item['wrc_plus'] = wrc_plus
                    
                    woba = game.get("wOBA%", LG_AVG_STATS[str(year)]['Bats']['woba'])
                    item['woba'] = woba

                    barrel_percent = game.get("Barrel%", LG_AVG_STATS[str(year)]['Bats']['barrel_percent'])
                    item['barrel_percent'] = barrel_percent

                    hard_hit = game.get("HardHit%", LG_AVG_STATS[str(year)]['Bats']['hard_hit'])
                    item['hard_hit'] = hard_hit

                    item['baserunning'] = game['wBSR']
                    item['date'] = game['gamedate']
                    item['scraped_at'] = datetime.now()
                    yield item
                
                elif type == 'W':
                    item = PitcherStat()
                    item['player_id'] = game['playerid']
                    item['name'] = game['PlayerName']
                    item['team'] = game['Team']
                    item['ip'] = game['IP']
                    item['dh'] = game['dh']

                    era = game.get("ERA", LG_AVG_STATS[str(year)]['Throws']['era'])
                    item['era'] = era
                    
                    k_percent = game.get("K%", LG_AVG_STATS[str(year)]['Throws']['k_percent'])
                    item['k_percent'] = k_percent

                    bb_percent = game.get("BB%", LG_AVG_STATS[str(year)]['Throws']['bb_percent'])
                    item['bb_percent'] = bb_percent

                    barrel_percent = game.get("Barrel%", LG_AVG_STATS[str(year)]['Throws']['barrel_percent'])
                    item['barrel_percent'] = barrel_percent

                    hard_hit = game.get("HardHit%", LG_AVG_STATS[str(year)]['Throws']['hard_hit'])
                    item['hard_hit'] = hard_hit

                    siera = game.get("SIERA", LG_AVG_STATS[str(year)]['Throws']['siera'])
                    item['siera'] = siera

                    fip = game.get("FIP", LG_AVG_STATS[str(year)]['Throws']['fip'])
                    item['fip'] = fip

                    item['date'] = game['gamedate']
                    item['scraped_at'] = datetime.now()
                    yield item
            else:
                continue



# For each date, fetch the unique player ID's and store them in a hashset
    # Also store position?, playerNameRoute
# For each player ID, go to the gamelog page for that player and then scrape/store their stats
    # Same fields as before, indexed on player ID, game date, dh field
        # dh field: 0: no dh, 1: 1st game, 2: 2nd game

# https://www.fangraphs.com/api/players/game-log?playerid=15640&position=OF&type=0&gds=2021-04-01&gde=2025-10-31&season=

# https://www.fangraphs.com/api/teams/lineup/games?teamid=14&season=2021
    # Get lineups for the whole season and extract player IDs, position (may need mappings), dh field
# vlad - 1B
# https://www.fangraphs.com/api/leaders/pitch-type?season=2021&startdate=2021-04-13&enddate=2021-04-13&pitchtype=FA%2CFC%2CFO%2CFS%2CSI%2CCH%2CSL%2CCU%2CKN%2CKC%2CSC%2CEP&position=bat&stands=&throws=&pteamids=&bteamids=&pitcherids=&batterids=&pitches=0&groupbylevel=player&groupbytime=game&includepitchpct=false&sortstat=Mov&sortdir=default&pagenum=1&pageitems=2000000000