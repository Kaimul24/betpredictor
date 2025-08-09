import scrapy
import json
from dotenv import load_dotenv
import os
from config import DATES, LG_AVG_STATS
from scrapers.items import BatterStat, PitcherStat, LineupItem, LineupPlayerItem
from datetime import datetime
from src.utils import normalize_names

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

                batting_order = p['BatOrder']

                if pos == 'P':
                    home_starter_id = p['playerId']
                    if year != '2021':
                        batting_order = None

                key = (id, year)
                if key in self.seen_player_season:
                    continue

                self.seen_player_season.add(id)

                player_item = LineupPlayerItem()
                player_item['date'] = game_info['gameDate'][:10]
                player_item['season'] = year
                player_item['dh'] = game_info['dh']
                player_item['team_id'] = game_info['teamid']
                player_item['team'] = game_info['Team']
                player_item['player_id'] = id
                player_item['position'] = pos
                player_item['batting_order'] = batting_order
                player_item['scraped_at'] = datetime.now()
                players_to_yield.append(player_item)

                url = GAME_LOG_URL.format(id, pos, year)

                yield scrapy.Request(
                    url,
                    cookies=self.cookies,
                    headers=self.json_headers,
                    cb_kwargs={"year": year, "id": id},
                    callback=self.parse_game_log,
                )

            lineup_item = LineupItem()
            lineup_item['team_id'] = game_info['teamid']
            lineup_item['team'] = game_info['Team']
            lineup_item['opposing_team_id'] = game_info['oppteamid']
            lineup_item['opposing_team'] = game_info['oppTeam']
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

    def parse_game_log(self, response, year, id):
        payload = json.loads(response.text)

        if payload == None:
            self.logger.warning(f'Payload is None for {response}\nYear: {year}\nID: {id}')
            return

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
                    item['normalized_player_name'] = normalize_names(game['PlayerName'])
                    item['team'] = game['Team']
                    item['batorder'] = game['BatOrder']
                    item['pos'] = game['Pos']
                    item['dh'] = game['dh']
                    item['ab'] = game['AB']
                    item['pa'] = game['PA']

                    ops = game.get("OPS", LG_AVG_STATS[str(year)]['Bats']['ops'])
                    item['ops'] = ops

                    babip = game.get("BABIP", LG_AVG_STATS[str(year)]['Bats']['babip'])
                    item['babip'] = babip

                    bb_k = game.get("BB/K", LG_AVG_STATS[str(year)]['Bats']['bb_k'])
                    item['bb_k'] = bb_k

                    wrc_plus = game.get("wRC+", 100)
                    item['wrc_plus'] = wrc_plus
                    
                    woba = game.get("wOBA", LG_AVG_STATS[str(year)]['Bats']['woba'])
                    item['woba'] = woba

                    barrel_percent = game.get("Barrel%", LG_AVG_STATS[str(year)]['Bats']['barrel_percent'])
                    item['barrel_percent'] = barrel_percent

                    hard_hit = game.get("HardHit%", LG_AVG_STATS[str(year)]['Bats']['hard_hit'])
                    item['hard_hit'] = hard_hit

                    ev = game.get("EV", LG_AVG_STATS[str(year)]['Bats']['ev'])
                    item['ev'] = ev

                    iso = game.get("ISO", LG_AVG_STATS[str(year)]['Bats']['iso'])
                    item['iso'] = iso

                    gb_fb = game.get("GB/FB", LG_AVG_STATS[str(year)]['Bats']['gb_fb'])
                    item['gb_fb'] = gb_fb

                    item['baserunning'] = game['wBSR']
                    
                    item['wraa'] = game['wRAA']

                    # wpa = game.get("WPA", LG_AVG_STATS[str(year)]['Bats']['wpa'])
                    item['wpa'] = game['WPA']

                    item['date'] = game['gamedate']
                    item['scraped_at'] = datetime.now()
                    item['season'] = year
                    yield item
                
                elif type == 'W':
                    item = PitcherStat()
                    item['player_id'] = game['playerid']
                    item['name'] = game['PlayerName']
                    item['normalized_player_name'] = normalize_names(game['PlayerName'])
                    item['team'] = game['Team']
                    item['dh'] = game['dh']
                    item['games'] = game['G']
                    item['gs'] = game['GS']
                    item['ip'] = game['IP']

                    item['runs'] = game['R']

                    era = game.get("ERA", LG_AVG_STATS[str(year)]['Throws']['era'])
                    item['era'] = era

                    babip = game.get("BABIP", LG_AVG_STATS[str(year)]['Throws']['babip'])
                    item['babip'] = babip
                    
                    k_percent = game.get("K%", LG_AVG_STATS[str(year)]['Throws']['k_percent'])
                    item['k_percent'] = k_percent

                    bb_percent = game.get("BB%", LG_AVG_STATS[str(year)]['Throws']['bb_percent'])
                    item['bb_percent'] = bb_percent

                    barrel_percent = game.get("Barrel%", LG_AVG_STATS[str(year)]['Throws']['barrel_percent'])
                    item['barrel_percent'] = barrel_percent

                    hard_hit = game.get("HardHit%", LG_AVG_STATS[str(year)]['Throws']['hard_hit'])
                    item['hard_hit'] = hard_hit

                    ev = game.get("EV", LG_AVG_STATS[str(year)]['Throws']['ev'])
                    item['ev'] = ev

                    hr_fb = game.get("HR/FB", LG_AVG_STATS[str(year)]['Throws']['hr_fb'])
                    item['hr_fb'] = hr_fb

                    siera = game.get("SIERA", LG_AVG_STATS[str(year)]['Throws']['siera'])
                    item['siera'] = siera

                    fip = game.get("FIP", LG_AVG_STATS[str(year)]['Throws']['fip'])
                    item['fip'] = fip

                    stuff = game.get("pb_stuff", 100)
                    item['stuff'] = stuff

                    item['ifbb'] = game['IFFB']
                    
                    wpa = game.get("WPA", 'UNKNOWN ERROR') #REMOVE LATER
                    if wpa == 'UNKNOWN ERROR':
                        print(f'\n***ERROR: WPA IS UNKNOWN***\n')
                    item['wpa'] = wpa

                    gmli = game.get("gmLI", LG_AVG_STATS[str(year)]['Throws']['gmli'])
                    item['gmli'] = gmli

                    item['date'] = game['gamedate']
                    item['scraped_at'] = datetime.now()
                    item['season'] = year
                    yield item
            else:
                continue