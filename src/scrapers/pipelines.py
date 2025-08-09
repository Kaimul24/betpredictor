# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import json
from pathlib import Path
from scrapers.items import BatterStat, PitcherStat, OddsItem, LineupItem, LineupPlayerItem, FRVItem
from data.database import get_database_manager, execute_query
from src.utils import normalize_names

class SqlitePipeline:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.ddl_path = Path(__file__).with_name("schema.sql")

    @classmethod
    def from_crawler(cls, crawler):
        db_path = crawler.settings.get('SQLITE_PATH', 'stats.db')
        return cls(db_path=db_path)

    def open_spider(self, spider):
        self.db_manager = get_database_manager(
            db_path=self.db_path,
            schema_path=self.ddl_path,
            max_connections=5
        )

        if spider.name == 'fg':
            tables_to_drop = ['lineup_players', 'lineups', 'batting_stats', 'pitching_stats', 'players']
            for table in tables_to_drop:
                execute_query(f"DROP TABLE IF EXISTS {table}", readonly=False)
        if spider.name == 'odds':
            execute_query("DROP TABLE IF EXISTS odds", readonly=False)

        if spider.name == 'fielding':
            execute_query("DROP TABLE IF EXISTS fielding", readonly=False)
        
        # Initialize schema
        self.db_manager.initialize_schema()

    def close_spider(self, spider):
        # No need to explicitly close connections - handled by database manager
        pass

    def process_item(self, item, spider):
        if not isinstance(item, (OddsItem, BatterStat, PitcherStat, LineupItem, LineupPlayerItem, FRVItem)):
            return item

        p = ItemAdapter(item)

        if isinstance(item, OddsItem):
            
            execute_query(
                """
                INSERT OR REPLACE INTO odds
                (game_date, game_datetime, away_team, home_team,
                 away_starter, home_starter,
                 away_starter_normalized,
                 home_starter_normalized,
                 away_score, home_score, winner,
                 sportsbook, away_odds, home_odds, season)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    p["date"],
                    p["game_datetime"],
                    p["away_team"],
                    p["home_team"],
                    p.get("away_starter"),
                    p.get("home_starter"),
                    p.get("away_starter_normalized"),
                    p.get("home_starter_normalized"),
                    p.get("away_score"),
                    p.get("home_score"),
                    p.get("winner"),
                    p["sportsbook"],
                    p["away_odds"],
                    p["home_odds"],
                    p['season'],
                ),
                readonly=False
            )
        elif isinstance(item, (BatterStat, PitcherStat)):
            pos = p.get("pos", 'P')
            normalized_name = normalize_names(p['name'])
            execute_query(
                "INSERT OR REPLACE INTO players(player_id, name, normalized_player_name, pos, current_team, last_updated) VALUES(?,?,?,?,?,?)",
                (p['player_id'], p['name'], normalized_name, pos, p['team'], p['scraped_at']),
                readonly=False
            )

            if isinstance(item, BatterStat):
                
                execute_query("""
                    INSERT OR REPLACE INTO batting_stats
                    (player_id, game_date, team, batorder, pos, dh, ab, pa, ops, babip, bb_k,
                     wrc_plus, woba, barrel_percent, hard_hit, ev, iso, gb_fb, baserunning, 
                     wraa, wpa, season, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (p['player_id'], p['date'], p['team'], p['batorder'], p['pos'], p['dh'], p['ab'], p['pa'],
                     p['ops'], p['babip'], p['bb_k'], p['wrc_plus'], p['woba'], p['barrel_percent'], p['hard_hit'], p['ev'],
                     p['iso'], p['gb_fb'], p['baserunning'], p['wraa'], p['wpa'], p['season'], p['scraped_at']),
                    readonly=False
                )

            if isinstance(item, PitcherStat):
                
                execute_query("""
                    INSERT OR REPLACE INTO pitching_stats
                    (player_id, game_date, team, dh, games, gs, era, babip, ip, runs, k_percent, 
                     bb_percent, barrel_percent, hard_hit, ev, hr_fb, siera, fip, stuff, 
                     ifbb, wpa, gmli, season, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (p['player_id'], p['date'], p['team'], p['dh'], p['games'], p['gs'], p['era'], p['babip'],
                     p['ip'], p['runs'], p['k_percent'], p['bb_percent'], p['barrel_percent'],
                     p['hard_hit'], p['ev'], p['hr_fb'], p['siera'], p['fip'], p['stuff'], p['ifbb'],
                     p['wpa'], p['gmli'], p['season'], p['scraped_at']),
                    readonly=False
                )
        elif isinstance(item, LineupItem):
            
            execute_query("""
                INSERT OR REPLACE INTO lineups
                (game_date, team_id, team, dh, opposing_team_id, opposing_team,
                 team_starter_id, opposing_starter_id, season, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (p['date'], p['team_id'], p['team'], p['dh'], p['opposing_team_id'], p['opposing_team'],
                 p['team_starter_id'], p['opposing_starter_id'], p['season'], p['scraped_at']),
                readonly=False
            )

        elif isinstance(item, LineupPlayerItem):
            
            execute_query("""
                INSERT OR REPLACE INTO lineup_players
                (game_date, team_id, team, dh, player_id, position, batting_order, season, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (p['date'], p['team_id'], p['team'], p['dh'], p['player_id'],
                 p['position'], p['batting_order'], p['season'], p['scraped_at']),
                readonly=False
            )

        elif isinstance(item, FRVItem):
            normalized_name = normalize_names(p['name'])
            execute_query("""
                INSERT OR REPLACE INTO fielding
                (name, normalized_player_name, season, frv, total_innings, innings_C, innings_1B, innings_2B, 
                 innings_3B, innings_SS, innings_LF, innings_CF, innings_RF)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (p['name'], normalized_name, p['season'], p['frv'], p['total_innings'], p['innings_C'],
                 p['innings_1B'], p['innings_2B'], p['innings_3B'], p['innings_SS'],
                 p['innings_LF'], p['innings_CF'], p['innings_RF']),
                readonly=False
            )

        return item

class DateRecorderPipeline:
    """Collect all requested and successfully scraped dates."""
    def open_spider(self, spider):
        self.requested = set()
        self.success = set()

    def process_item(self, item, spider):
        date = item.get("date")
        if date:
            self.success.add(date)
        return item

    def close_spider(self, spider):
        requested = sorted(getattr(spider, 'requested_dates', []))
        scraped = sorted(self.success)

        manifest = {
            "requested": requested,
            "scraped":   scraped,
            "missing":   sorted(set(requested) - set(scraped)),
        }

        with open("dates_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

