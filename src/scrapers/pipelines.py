# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import sqlite3, json
from pathlib import Path
from scrapers.items import BatterStat, PitcherStat, OddsItem, LineupItem, LineupPlayerItem

class SqlitePipeline:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.ddl_path = Path(__file__).with_name("schema.sql")

    @classmethod
    def from_crawler(cls, crawler):
        db_path = crawler.settings.get('SQLITE_PATH', 'stats.db')
        return cls(db_path=db_path)

    def open_spider(self, spider):
        # Ensure the directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.cur = self.conn.cursor()
        
        if spider.name == 'fg':
            tables_to_drop = ['lineup_players', 'lineups', 'batting_stats', 'pitching_stats', 'players']
            for table in tables_to_drop:
                self.cur.execute(f"DROP TABLE IF EXISTS {table}")
            self.conn.commit()
        
        self.cur.executescript(self.ddl_path.read_text())


    def close_spider(self, spider):
        self.conn.commit()
        self.conn.close()

    def process_item(self, item, spider):
        if not isinstance(item, (OddsItem, BatterStat, PitcherStat, LineupItem, LineupPlayerItem)):
            return item

        p = ItemAdapter(item)

        if isinstance(item, OddsItem):
            self.cur.execute(
                """
                INSERT OR REPLACE INTO odds
                (game_date, away_team, home_team,
                 away_starter, home_starter,
                 away_score, home_score, winner,
                 sportsbook, away_odds, home_odds)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    p["date"],
                    p["away_team"],
                    p["home_team"],
                    p.get("away_starter"),
                    p.get("home_starter"),
                    p.get("away_score"),
                    p.get("home_score"),
                    p.get("winner"),
                    p["sportsbook"],
                    p["away_odds"],
                    p["home_odds"],
                ),
            )
        elif isinstance(item, (BatterStat, PitcherStat)):
            self.cur.execute(
                "INSERT OR REPLACE INTO players(player_id, name, current_team, last_updated) VALUES(?,?,?,?)",
                (p['player_id'], p['name'], p['team'], p['scraped_at'])
            )

            if isinstance(item, BatterStat):
                self.cur.execute("""
                    INSERT OR REPLACE INTO batting_stats
                    (player_id, game_date, team, dh, ab, pa, ops, bb_k,
                     wrc_plus, woba, barrel_percent, hard_hit, baserunning, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (p['player_id'], p['date'], p['team'], p['dh'], p['ab'], p['pa'], p['ops'],
                     p['babip'], p['bb_k'], p['wrc_plus'], p['woba'], p['barrel_percent'], p['hard_hit'],
                     p['baserunning'], p['scraped_at'])
                )

            if isinstance(item, PitcherStat):
                self.cur.execute("""
                    INSERT OR REPLACE INTO pitching_stats
                    (player_id, game_date, team, dh, era, ip, k_percent, bb_percent,
                     barrel_percent, hard_hit, siera, fip, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (p['player_id'], p['date'], p['team'], p['dh'], p['era'], p['ip'],
                     p['babip'], p['k_percent'], p['bb_percent'], p['barrel_percent'],
                     p['hard_hit'], p['siera'], p['fip'], p['scraped_at'])
                )
        elif isinstance(item, LineupItem):
            self.cur.execute("""
                INSERT OR REPLACE INTO lineups
                (game_date, team_id, dh, opposing_team_id, 
                 team_starter_id, opposing_starter_id, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (p['date'], p['team_id'], p['dh'], p['opposing_team_id'],
                 p['team_starter_id'], p['opposing_starter_id'], p['scraped_at'])
            )

        elif isinstance(item, LineupPlayerItem):
            self.cur.execute("""
                INSERT OR REPLACE INTO lineup_players
                (game_date, team_id, dh, player_id, position, batting_order, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (p['date'], p['team_id'], p['dh'], p['player_id'],
                 p['position'], p['batting_order'], p['scraped_at'])
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


