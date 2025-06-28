from datetime import date
from pathlib import Path
import sqlite3

PROJECT_ROOT = Path(__file__).parent.parent

DATABASE_PATH = PROJECT_ROOT / 'data' / 'mlb_stats.sqlite'
SCHEMA_PATH = PROJECT_ROOT / 'src' / 'scrapers' / 'schema.sql'

def connect_database():
    """Connect to the existing database without modifying schema. Returns connection and cursor."""
    DATABASE_PATH.parent.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(str(DATABASE_PATH))
    cursor = conn.cursor()
    
    with open(SCHEMA_PATH, 'r') as f:
        schema = f.read()
    cursor.executescript(schema)
    
    conn.commit()
    return conn, cursor

DATES = {
    '2021': [date(2021, 4, 1), date(2021, 10, 4)],
    '2022': [date(2022, 4, 7), date(2022, 10, 6)],
    '2023': [date(2023, 3, 30), date(2023, 10, 2)],
    '2024': [date(2024, 3, 28), date(2024, 10, 1)]
}

TEAM_ABBR_MAP = {
    'Arizona Diamondbacks': 'ARI',
    'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC',
    'Chicago White Sox': 'CWS',
    'Cincinnati Reds': 'CIN',
    'Cleveland Guardians': 'CLE',
    'Cleveland Indians': 'CLE',
    'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU',
    'Kansas City Royals': 'KC',
    'Los Angeles Angels': 'LAA',
    'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA',
    'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM',
    'New York Yankees': 'NYY',
    'Oakland Athletics': 'OAK',
    'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SD',
    'San Francisco Giants': 'SF',
    'Seattle Mariners': 'SEA',
    'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TB',
    'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR',
    'Washington Nationals': 'WSH'
}