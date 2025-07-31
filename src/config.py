from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATABASE_PATH = PROJECT_ROOT / 'data' / 'mlb_stats.sqlite'
SCHEMA_PATH = PROJECT_ROOT / 'src' / 'scrapers' / 'schema.sql'

DATES = {
    '2021': [date(2021, 4, 1), date(2021, 10, 4)],
    '2022': [date(2022, 4, 7), date(2022, 10, 6)],
    '2023': [date(2023, 3, 30), date(2023, 10, 2)],
    '2024': [date(2024, 3, 28), date(2024, 10, 1)]
}

LG_AVG_STATS = {
    "2021": {"Bats": {"ops": 0.7278624, "babip": 0.291664854, "bb_k": 0.374753826, "woba": 0.31442644240601775, "barrel_percent": 0.07911897, "hard_hit": 0.38520605, "ev": 88.76291668581086, "iso": 0.166955867, "gb_fb": 1.176134806}, "Throws": {"era": 4.265892667856131, "babip": 0.2895822900657906, "k_percent": 0.23179901, "bb_percent": 0.08686756, "barrel_percent": 0.07911897, "hard_hit": 0.38520605, "siera": 4.175190307395692, "fip": 4.265892485031086, "ev": 88.76291679304866, "hr_fb": 0.135720157, "gmli": 1.0584131599484485}},
    "2022": {"Bats": {"ops": 0.706412709, "babip": 0.290404678, "bb_k": 0.363937077, "woba": 0.30974585067181093, "barrel_percent": 0.07502755, "hard_hit": 0.38169228, "ev": 88.57476170256407, "iso": 0.152148778, "gb_fb": 1.153070137}, "Throws": {"era": 3.96832714563721, "babip": 0.28927354229974983, "k_percent": 0.22417771, "bb_percent": 0.08158658, "barrel_percent": 0.07502755, "hard_hit": 0.38169228, "siera": 3.8756488351581293, "fip": 3.9683269469843023, "ev": 88.57476181142745, "hr_fb": 0.113874574, "gmli": 1.057056432586794}},
    "2023": {"Bats": {"ops": 0.734292472, "babip": 0.296522719, "bb_k": 0.378056066, "woba": 0.31837469936833196, "barrel_percent": 0.08060002, "hard_hit": 0.39204536, "ev": 89.00106155112319, "iso": 0.165772604, "gb_fb": 1.134658315}, "Throws": {"era": 4.331714250563033, "babip": 0.2952085900964022, "k_percent": 0.22727915, "bb_percent": 0.08592426, "barrel_percent": 0.08060002, "hard_hit": 0.39204536, "siera": 4.237841443447046, "fip": 4.331714183169491, "ev": 89.00106150666907, "hr_fb": 0.127180909, "gmli": 1.0809939454776771}},
    "2024": {"Bats": {"ops": 0.711329941, "babip": 0.290537456, "bb_k": 0.362380755, "woba": 0.310181047531053, "barrel_percent": 0.07797881, "hard_hit": 0.38651521, "ev": 88.83302249496766, "iso": 0.155931748, "gb_fb": 1.108267212}, "Throws": {"era": 4.07894181840083,  "babip": 0.28918805216659654, "k_percent": 0.22580009, "bb_percent": 0.08182561, "barrel_percent": 0.07797881, "hard_hit": 0.38651521, "siera": 3.9892671273186298, "fip": 4.078941491380367, "ev": 88.8330223533932, "hr_fb": 0.116308335, "gmli": 1.0586196697086832}}
}

POSITION_MAP = {
    '1': 'P',
    '2': 'C',
    '3': '1B',
    '4': '2B',
    '5': '3B',
    '6': 'SS',
    '7': 'LF',
    '8': 'CF',
    '9': 'RF'
}

TEAM_ABBR_MAP = {
    'Arizona Diamondbacks': 'ARI',
    'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC',
    'Chicago White Sox': 'CHW',
    'Cincinnati Reds': 'CIN',
    'Cleveland Guardians': 'CLE',
    'Cleveland Indians': 'CLE',
    'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU',
    'Kansas City Royals': 'KCR',
    'Los Angeles Angels': 'LAA',
    'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA',
    'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM',
    'New York Yankees': 'NYY',
    'Oakland Athletics': 'ATH',
    'Athletics': 'ATH', # MAY NEED TO CHANGE
    'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SDP',
    'San Francisco Giants': 'SFG',
    'Seattle Mariners': 'SEA',
    'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TBR',
    'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR',
    'Washington Nationals': 'WSN'
}

ODDS_TEAM_ABBR_MAP = {
    'AZ': 'ARI',
    'KC': 'KCR',
    'OAK': 'ATH',
    'SD': 'SDP',
    'SF': 'SFG',
    'TB': 'TBR',
    'WAS': 'WSN'
}

TEAM_ID_MAP = {
    
}