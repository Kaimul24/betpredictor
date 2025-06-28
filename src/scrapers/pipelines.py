# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import sqlite3, json, os
from pathlib import Path
from scrapers.items import BatterStat, PitcherStat, OddsItem

class OddsPipeline:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.ddl_path = Path(__file__).with_name("schema.sql")

    @classmethod
    def from_crawler(cls, crawler):
        db_path = crawler.settings.get('SQLITE_PATH', 'stats.db')
        return cls(db_path=db_path)

    def open_spider(self, spider):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.cur = self.conn.cursor()

        # Drop and recreate odds table to ensure clean data
        self.cur.execute("DROP TABLE IF EXISTS odds")
        self.cur.executescript(self.ddl_path.read_text())

    def close_spider(self, spider):
        self.conn.commit()
        self.conn.close()

    def process_item(self, item, spider):
        if not isinstance(item, OddsItem):
            return item

        p = ItemAdapter(item)

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

        return item

class StatsPipeline:
    def __init__(self, db_path):
        self.db_path = db_path
        self.ddl_path = Path(__file__).with_name("schema.sql")

    @classmethod
    def from_crawler(cls, crawler):
        db_path = crawler.settings.get('SQLITE_PATH', 'stats.db')
        return cls(db_path=db_path)
    

    def open_spider(self, spider):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.cur = self.conn.cursor()
        self.cur.executescript(self.ddl_path.read_text())

    def close_spider(self, spider):
        self.conn.commit()
        self.conn.close()

    def process_item(self, item, spider):
        if not isinstance(item, BatterStat) or isinstance(item, PitcherStat):
            return item

        p = ItemAdapter(item)
        
        self.cur.execute(
            "INSERT OR IGNORE INTO players(player_id, name, team) VALUES(?,?,?)",
            (p['player_id'], p['name'], p['team'])
        )

        if isinstance(item, BatterStat):
            self.cur.execute("""
                INSERT OR REPLACE INTO batting_stats
                (player_id, game_date, team, games, ab, pa, ops, bb_k,
                 wrc_plus, barrel_percent, hard_hit, war, baserunning, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (p['player_id'], p['date'], p['team'], p['games'], p['ab'], p['pa'], p['ops'],
                 p['bb_k'], p['wrc_plus'], p['barrel_percent'], p['hard_hit'],
                 p['war'], p['baserunning'], p['scraped_at'])
            )

        if isinstance(item, PitcherStat):
            self.cur.execute("""
                INSERT OR REPLACE INTO pitching_stats
                (player_id, game_date, team, games, era, ip, k_percent, bb_percent,
                 barrel_percent, hard_hit, war, siera, fip, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (p['player_id'], p['date'], p['team'], p['games'], p['era'], p['ip'],
                 p['k_percent'], p['bb_percent'], p['barrel_percent'],
                 p['hard_hit'], p['war'], p['siera'], p['fip'], p['scraped_at'])
            )

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
        requested = sorted(spider.requested_dates)
        scraped = sorted(self.success)

        manifest = {
            "requested": requested,
            "scraped":   scraped,
            "missing":   sorted(set(requested) - set(scraped)),
        }
        with open("dates_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

