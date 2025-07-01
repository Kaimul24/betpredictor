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

LG_AVG_STATS = {
    "2021": {"Bats": {"ops": 0.7278624, "bb_k": 0.374753826, "woba": 0.31442644240601775, "barrel_percent": 0.07911897, "hard_hit": 0.38520605}, "Throws": {"era": 4.265892667856131, "k_percent": 0.23179901, "bb_percent": 0.08686756, "barrel_percent": 0.07911897, "hard_hit": 0.38520605, "siera": 4.175190307395692, "fip": 4.265892485031086}},
    "2022": {"Bats": {"ops": 0.706412709, "bb_k": 0.363937077, "woba": 0.30974585067181093, "barrel_percent": 0.07502755, "hard_hit": 0.38169228}, "Throws": {"era": 3.96832714563721, "k_percent": 0.22417771, "bb_percent": 0.08158658, "barrel_percent": 0.07502755, "hard_hit": 0.38169228, "siera": 3.8756488351581293, "fip": 3.9683269469843023}},
    "2023": {"Bats": {"ops": 0.734292472, "bb_k": 0.378056066, "woba": 0.31837469936833196, "barrel_percent": 0.08060002, "hard_hit": 0.39204536}, "Throws": {"era": 4.331714250563033, "k_percent": 0.22727915, "bb_percent": 0.08592426, "barrel_percent": 0.08060002, "hard_hit": 0.39204536, "siera": 4.237841443447046, "fip": 4.331714183169491}},
    "2024": {"Bats": {"ops": 0.711329941, "bb_k": 0.362380755, "woba": 0.310181047531053, "barrel_percent": 0.07797881, "hard_hit": 0.38651521}, "Throws": {"era": 4.07894181840083, "k_percent": 0.22580009, "bb_percent": 0.08182561, "barrel_percent": 0.07797881, "hard_hit": 0.38651521, "siera": 3.9892671273186298, "fip": 4.078941491380367}}
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