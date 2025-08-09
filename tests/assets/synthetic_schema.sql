-- Synthetic test schema for loader tests
-- Contains only the essential tables and columns needed by the loaders

CREATE TABLE IF NOT EXISTS schedule (
  game_id       TEXT PRIMARY KEY,
  game_date     TEXT NOT NULL,
  game_datetime TEXT,
  season        INTEGER,
  away_team     TEXT NOT NULL,
  home_team     TEXT NOT NULL,
  dh            INTEGER,
  venue_name    TEXT,
  venue_id      INTEGER,
  status        TEXT,
  away_probable_pitcher TEXT,
  home_probable_pitcher TEXT,
  away_starter_normalized TEXT,
  home_starter_normalized TEXT,
  wind          TEXT,
  condition     TEXT,
  temp          INTEGER,
  away_score    INTEGER,
  home_score    INTEGER,
  winning_team  TEXT,
  losing_team   TEXT,
  scraped_at    TEXT
);

CREATE TABLE IF NOT EXISTS odds (
  game_date     TEXT,
  game_datetime TEXT,
  away_team     TEXT,
  home_team     TEXT,
  away_starter  TEXT,
  home_starter  TEXT,
  away_starter_normalized TEXT,
  home_starter_normalized TEXT,
  away_score    INTEGER,
  home_score    INTEGER,
  winner        TEXT,
  sportsbook    TEXT,
  away_odds     REAL,
  home_odds     REAL,
  season        INTEGER NOT NULL,
  PRIMARY KEY (game_date, away_team, home_team, sportsbook)
);

CREATE TABLE IF NOT EXISTS players (
  player_id TEXT PRIMARY KEY,      
  name      TEXT,
  normalized_player_name TEXT,
  pos       TEXT,
  current_team TEXT,
  last_updated TEXT
);

CREATE TABLE IF NOT EXISTS batting_stats (
  player_id       TEXT, 
  game_date       TEXT,
  team            TEXT,
  batorder        TEXT,
  pos             TEXT,
  dh              INTEGER,
  ab              INTEGER,
  pa              INTEGER,
  ops             REAL,
  babip           REAL,
  bb_k            REAL,
  wrc_plus        REAL,
  woba            REAL,
  barrel_percent  REAL,
  hard_hit        REAL,
  ev              REAL,
  iso             REAL,
  gb_fb           REAL,
  baserunning     REAL,
  wraa            REAL,
  wpa             REAL,
  season          INTEGER NOT NULL,
  scraped_at      TEXT,
  PRIMARY KEY (player_id, game_date, dh),
  FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS pitching_stats (
  player_id       TEXT,
  game_date       TEXT,
  team            TEXT,
  dh              INTEGER,
  games           INTEGER,
  gs              INTEGER,
  era             REAL,
  babip           REAL,
  ip              REAL,
  runs            INTEGER,
  k_percent       REAL,
  bb_percent      REAL,
  barrel_percent  REAL,
  hard_hit        REAL,
  ev              REAL,
  hr_fb           REAL,
  siera           REAL,
  fip             REAL,
  stuff           INTEGER,
  ifbb            INTEGER,
  wpa             REAL,
  gmli            REAL,
  season          INTEGER NOT NULL,
  scraped_at      TEXT,
  PRIMARY KEY (player_id, game_date, dh),
  FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS lineups (
  game_date     TEXT NOT NULL,
  team_id       INTEGER NOT NULL,
  team          TEXT,
  dh            INTEGER NOT NULL,
  opposing_team_id INTEGER NOT NULL,
  opposing_team   TEXT,
  team_starter_id TEXT,
  opposing_starter_id TEXT,
  season        INTEGER NOT NULL,
  scraped_at    TEXT,
  PRIMARY KEY (game_date, team_id, dh)
);

CREATE TABLE IF NOT EXISTS lineup_players (
  game_date     TEXT NOT NULL,
  team_id       INTEGER NOT NULL,
  team          TEXT,
  dh            INTEGER NOT NULL,
  player_id     TEXT NOT NULL,
  position      TEXT NOT NULL,
  batting_order INTEGER,
  season        INTEGER NOT NULL,
  scraped_at    TEXT,
  PRIMARY KEY (game_date, team_id, dh, player_id)
);

CREATE TABLE IF NOT EXISTS fielding (
  name            TEXT NOT NULL,
  normalized_player_name TEXT,
  season          INTEGER NOT NULL,
  frv             REAL,
  total_innings   REAL,
  innings_C       REAL,
  innings_1B      REAL,
  innings_2B      REAL,
  innings_3B      REAL,
  innings_SS      REAL,
  innings_LF      REAL,
  innings_CF      REAL,
  innings_RF      REAL,
  PRIMARY KEY (name, season)
);

CREATE TABLE IF NOT EXISTS rosters (
  game_date     TEXT NOT NULL,
  season        TEXT,
  team          TEXT NOT NULL,
  player_name   TEXT NOT NULL,
  normalized_player_name TEXT,
  position      TEXT NOT NULL,
  status        TEXT NOT NULL,
  scraped_at    TEXT,
  PRIMARY KEY (game_date, player_name, team)
);

