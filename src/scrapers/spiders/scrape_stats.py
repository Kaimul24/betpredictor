import scrapy
import json
from dotenv import load_dotenv
import os
from config import DATES, LG_AVG_STATS, TEAM_ABBR_MAP
from scrapers.items import BatterStat, PitcherStat, LineupItem, LineupPlayerItem
from datetime import datetime
from src.utils import normalize_names

load_dotenv()
PITCHERS_URL = os.getenv("PITCHERS_URL")
BATTERS_URL = os.getenv("BATTERS_URL")
GAME_LOG_URL = os.getenv("GAME_URL")

class fgSpider(scrapy.Spider):
    name = "stats"
    
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

        for year in DATES.keys():
            if int(year) == 2021:
                pit_url = PITCHERS_URL.format(year, year)
                yield scrapy.Request(
                    pit_url,
                    cookies=self.cookies,
                    headers=self.json_headers,
                    callback=self.get_player_ids,
                    cb_kwargs={"year": year, "stats_type": "pitching"},
                )

                bat_url = BATTERS_URL.format(year, year)
                yield scrapy.Request(
                    bat_url,
                    cookies=self.cookies,
                    headers=self.json_headers,
                    callback=self.get_player_ids,
                    cb_kwargs={"year": year, "stats_type": "batting"},
                )

    def get_player_ids(self, response, year, stats_type):
        payload = json.loads(response.text)

        if payload == [] or payload == None:
            self.logger.warning(f"No roster data found for {year}")
            self.logger.warning(f'Payload is None for {response}')
            return


        data = payload['data']
        for player in data:
            player_id = player['playerid']
            mlb_id = player['xMLBAMID']
            
            # Use appropriate position based on stats type
            if stats_type == "pitching":
                pos = "P"  # Force pitching position for pitchers API
            else:
                pos = "all"  # Use "all" for batting stats
                
            url = GAME_LOG_URL.format(player_id, pos, year)

            yield scrapy.Request(
                    url,
                    cookies=self.cookies,
                    headers=self.json_headers,
                    cb_kwargs={"year": year, "id": player_id, "mlb_id": mlb_id},
                    callback=self.parse_game_log,
                )
            
    def parse_game_log(self, response, year, id, mlb_id):
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
            game_flag = game['G']
            # if events == 0:
            #     print(f"EVENTS is 0 for {id}")
            #     print(f"{game['PlayerName']}, {game['gamedate']}")
            #     print(f"{game.keys()}\n")

            if type == 'G':
                item = BatterStat()
                item['player_id'] = game['playerid']
                item['mlb_id'] = mlb_id
                item['name'] = game['PlayerName']
                item['normalized_player_name'] = normalize_names(game['PlayerName'])
                item['team'] = TEAM_ABBR_MAP.get(game['Team'], game['Team'])
                item['batorder'] = game['BatOrder']
                item['pos'] = game['Pos']
                item['dh'] = game['dh']
                item['ab'] = game['AB']
                item['pa'] = game['PA']

                bip = game.get("bipCount", 0)
                item['bip'] = bip

                ops = game.get("OPS", 0.0)
                item['ops'] = ops

                babip = game.get("BABIP", 0.0)
                item['babip'] = babip

                bb_k = game.get("BB/K", 0.0)
                item['bb_k'] = bb_k

                wrc_plus = game.get("wRC+", 100)
                item['wrc_plus'] = wrc_plus
                
                woba = game.get("wOBA", 0.0)
                item['woba'] = woba

                barrel_percent = game.get("Barrel%", 0.0)
                item['barrel_percent'] = barrel_percent

                hard_hit = game.get("HardHit%", 0.0)
                item['hard_hit'] = hard_hit

                ev = game.get("EV", 0.0)
                item['ev'] = ev

                iso = game.get("ISO", 0.0)
                item['iso'] = iso

                gb_fb = game.get("GB/FB", 0.0)
                item['gb_fb'] = gb_fb

                baserunning = game.get('wBSR', 0.0)
                item['baserunning'] = baserunning
                
                wraa = game.get('wRAA', 0)
                item['wraa'] = wraa

                wpa = game.get("WPA", 0.0)
                item['wpa'] = wpa

                item['date'] = game['gamedate']
                item['scraped_at'] = datetime.now()
                item['season'] = year
                yield item
            
            elif type == 'W':
                item = PitcherStat()
                item['player_id'] = game['playerid']
                item['mlb_id'] = mlb_id
                item['name'] = game['PlayerName']
                item['normalized_player_name'] = normalize_names(game['PlayerName'])
                item['team'] = TEAM_ABBR_MAP.get(game['Team'], game['Team'])
                item['dh'] = game['dh']
                item['games'] = game['G']
                item['gs'] = game['GS']

                ip = game.get("IP", 0.0)
                item['ip'] = ip

                tbf = game.get("TBF", 0)
                item['tbf'] = tbf

                bip = game.get('bipCount', 0)
                item['bip'] = bip
                
                runs = game.get('R', 0)
                item['runs'] = runs

                era = game.get("ERA", 0.0)
                item['era'] = era

                babip = game.get("BABIP", 0.0)
                item['babip'] = babip
                
                k_percent = game.get("K%", 0.0)
                item['k_percent'] = k_percent

                bb_percent = game.get("BB%", 0.0)
                item['bb_percent'] = bb_percent

                barrel_percent = game.get("Barrel%", 0.0)
                item['barrel_percent'] = barrel_percent

                hard_hit = game.get("HardHit%", 0.0)
                item['hard_hit'] = hard_hit

                ev = game.get("EV", 0.0)
                item['ev'] = ev

                hr_fb = game.get("HR/FB", 0.0)
                item['hr_fb'] = hr_fb

                siera = game.get("SIERA", 0.0)
                item['siera'] = siera

                fip = game.get("FIP", 0.0)
                item['fip'] = fip

                stuff = game.get("sp_stuff", 100) ### CHANGE TO sp_stuff
                item['stuff'] = stuff
                
                iffb = game.get("IFFB", 0)
                item['iffb'] = iffb
                
                wpa = game.get("WPA", 0.0)

                item['wpa'] = wpa

                gmli = game.get("gmLI", 0.0)
                item['gmli'] = gmli

                item['fa_percent'] = game.get('pfxFA%', 0.0)
                item['fc_percent'] = game.get('pfxFC%', 0.0)
                item['si_percent'] = game.get('pfxSI%', 0.0)

                item['fa_velo'] = game.get('pivFA', 0.0)
                item['fc_velo'] = game.get('pivFC', 0.0)
                item['si_velo'] = game.get('pivSI', 0.0)

                item['date'] = game['gamedate']
                item['scraped_at'] = datetime.now()
                item['season'] = year
                yield item
