import scrapy
import json
from dotenv import load_dotenv
import os
from config import DATES, LG_AVG_STATS, TEAM_ABBR_MAP
from scrapers.items import BatterStat, PitcherStat, LineupItem, LineupPlayerItem
from datetime import datetime
from src.utils import normalize_names

# USE LEADERBOARDS ENDPOINT TO GET ALL PLAYER IDS, THEN GET EACH PLAYER'S GAME LOGS FOR THE YEAR
# TURN THIS SCRIPT INTO LINEUPS
# FROM LEADERBOARDS ENDPOINT, ALSO GET MLB PLAYERID FOR EASIER JOINS BETWEEN SOURCES

load_dotenv()
TEAM_URL = os.getenv("TEAM_URL")
GAME_LOG_URL = os.getenv("GAME_URL")

class fgSpider(scrapy.Spider):
    name = "lineups"
    
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
            callback=self.after_handshake
        )
    
    async def after_handshake(self, response):
        page = response.meta["playwright_page"]
        cookies = await page.context.cookies()
        self.cookies = {c["name"]: c["value"] for c in cookies}
        await page.close()

        self.seen_player_season = set()

        for team_id in range(1, 31):
            for year in DATES.keys():
                if int(year) == 2021:
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

        lineups_to_yield = []
        players_to_yield = []

        if payload == [] or payload == None:
            self.logger.warning(f"No roster data found for {year}")
            self.logger.warning(f'Payload is None for {response}')
            return

        for g in payload:
            lineup = g['lineupInfo']
            game_info = g['gameInfo'][0]
            home_starter_id = None

            for p in lineup:
                id = p['playerId']
                pos = p['position']
                gs = p['GS']

                batting_order = p['BatOrder']

                if gs == 1:
                    home_starter_id = p['playerId']
                    if year == '2021' and batting_order == 0:
                        batting_order = None

                key = (id, year)
                if key in self.seen_player_season:
                    continue

                self.seen_player_season.add(id)
                team = game_info['Team']
                team_abbr = TEAM_ABBR_MAP.get(team, None)
                if team_abbr == None:
                    raise ValueError(f"Bug in team abbr map: {team}")
                
                opposing_team = game_info['oppTeam']
                opposing_team_abbr = TEAM_ABBR_MAP.get(opposing_team, None)
                if opposing_team_abbr == None:
                    raise ValueError(f"Bug in team opp abbr map: {opposing_team}")

                player_item = LineupPlayerItem()
                player_item['date'] = game_info['gameDate'][:10]
                player_item['season'] = year
                player_item['dh'] = game_info['dh']
                player_item['team_id'] = game_info['oppteamid']
                player_item['team'] = team_abbr
                player_item['opposing_team_id'] = game_info['teamid']
                player_item['opposing_team'] = opposing_team_abbr
                player_item['player_id'] = id
                player_item['position'] = pos
                player_item['batting_order'] = batting_order
                player_item['scraped_at'] = datetime.now()
                players_to_yield.append(player_item)

            lineup_item = LineupItem()
            lineup_item['team_id'] = game_info['oppteamid']
            lineup_item['team'] = TEAM_ABBR_MAP.get(game_info['Team'], None)
            lineup_item['opposing_team_id'] = game_info['teamid']
            lineup_item['opposing_team'] = TEAM_ABBR_MAP.get(game_info['oppTeam'], None)
            lineup_item['date'] = game_info['gameDate'][:10]
            lineup_item['season'] = year
            lineup_item['dh'] = game_info['dh']
            lineup_item['team_starter_id'] = home_starter_id
            lineup_item['opposing_starter_id'] = game_info['oppSP']
            lineup_item['scraped_at'] = datetime.now()
            lineups_to_yield.append(lineup_item)

        for lineup_ in lineups_to_yield:
            yield lineup_

        for player in players_to_yield:
            yield player
