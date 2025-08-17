CREATE TABLE IF NOT EXISTS players (
  player_id TEXT PRIMARY KEY,      
  name      TEXT,
  normalized_player_name TEXT,
  pos       TEXT,
  current_team TEXT,
  last_updated TEXT
);

CREATE TABLE IF NOT EXISTS rosters (
  game_date TEXT NOT NULL,
  team TEXT NOT NULL,
  season TEXT,
  player_name TEXT NOT NULL,
  normalized_player_name TEXT,
  position TEXT NOT NULL,
  status TEXT NOT NULL,
  scraped_at TEXT,
  PRIMARY KEY (game_date, player_name, team)
);

CREATE INDEX IF NOT EXISTS idx_rosters_team_date
  ON rosters(game_date, team);

CREATE TABLE IF NOT EXISTS schedule (
  game_id       TEXT PRIMARY KEY,
  game_date     TEXT NOT NULL,
  game_datetime TEXT,
  day_night_game TEXT,
  season        INTEGER NOT NULL,
  away_team     TEXT NOT NULL,
  home_team     TEXT NOT NULL,
  dh            INTEGER,
  venue_name    TEXT,
  venue_id      INTEGER,
  venue_elevation INTEGER,
  venue_timezone TEXT,
  venue_gametime_offset INTEGER,
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

CREATE INDEX IF NOT EXISTS idx_schedule_year
  ON schedule(season);

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
  away_opening_odds     REAL,
  home_opening_odds     REAL,
  away_current_odds     REAL,
  home_current_odds     REAL,
  season          INTEGER,
  PRIMARY KEY (game_date, away_team, home_team, sportsbook)
);

CREATE INDEX IF NOT EXISTS idx_odds_date
  ON odds(game_date);

CREATE INDEX IF NOT EXISTS idx_odds_year
  ON odds(season);

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

CREATE INDEX IF NOT EXISTS idx_batting_team_date
  ON batting_stats(team, game_date);

CREATE INDEX IF NOT EXISTS idx_pitching_team_date
  ON pitching_stats(team, game_date);

CREATE INDEX IF NOT EXISTS idx_batting_player_date
  ON batting_stats(player_id, game_date);

CREATE INDEX IF NOT EXISTS idx_pitching_player_date
  ON pitching_stats(player_id, game_date);

CREATE INDEX IF NOT EXISTS idx_batting_team_player_date
  ON batting_stats(team, player_id, game_date);

CREATE INDEX IF NOT EXISTS idx_pitching_team_player_date
  ON pitching_stats(team, player_id, game_date);

CREATE INDEX IF NOT EXISTS idx_batting_year_team
  ON batting_stats(season, team);

CREATE INDEX IF NOT EXISTS idx_pitching_year_team
  ON pitching_stats(season, team);

CREATE TABLE IF NOT EXISTS lineups (
  game_date     TEXT NOT NULL,
  team_id       INTEGER NOT NULL,
  team          TEXT NOT NULL,
  dh            INTEGER NOT NULL,
  opposing_team_id INTEGER NOT NULL,
  opposing_team   TEXT,
  team_starter_id TEXT,
  opposing_starter_id TEXT,
  season        INTEGER NOT NULL,
  scraped_at    TEXT,
  PRIMARY KEY (game_date, team, dh)
);

CREATE TABLE IF NOT EXISTS lineup_players (
  game_date     TEXT NOT NULL,
  team_id       INTEGER NOT NULL,
  team          TEXT NOT NULL,
  opposing_team_id INTEGER,
  opposing_team TEXT,
  dh            INTEGER NOT NULL,
  player_id     TEXT NOT NULL,
  position      TEXT NOT NULL,
  batting_order INTEGER,
  season          INTEGER NOT NULL,
  scraped_at    TEXT,
  PRIMARY KEY (game_date, team, dh, player_id)
);

CREATE INDEX IF NOT EXISTS idx_lineups_date
  ON lineups(game_date);

CREATE INDEX IF NOT EXISTS idx_lineups_team_date
  ON lineups(game_date, team);

CREATE INDEX IF NOT EXISTS idx_lineup_players_date
  ON lineup_players(game_date);

CREATE INDEX IF NOT EXISTS idx_lineup_players_team_date
  ON lineup_players(game_date, team);

CREATE INDEX IF NOT EXISTS idx_lineup_players_player
  ON lineup_players(player_id, game_date);

CREATE INDEX IF NOT EXISTS idx_lineup_players_game
  ON lineup_players(game_date, dh, team);

CREATE INDEX IF NOT EXISTS idx_lineup_players_position
  ON lineup_players(position, game_date);

CREATE INDEX IF NOT EXISTS idx_lineup_players_batting_order
  ON lineup_players(batting_order, game_date);

CREATE INDEX IF NOT EXISTS idx_lineups_year
  ON lineups(season);

CREATE INDEX IF NOT EXISTS idx_lineup_players_year
  ON lineup_players(season);

CREATE TABLE IF NOT EXISTS fielding (
  name            TEXT NOT NULL,
  normalized_player_name TEXT,
  season          INTEGER NOT NULL,
  month           INTEGER,
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
  PRIMARY KEY (name, month, season)
);

CREATE INDEX IF NOT EXISTS idx_fielding_year
  ON fielding(season);

CREATE INDEX IF NOT EXISTS idx_fielding_normalized_name
  ON fielding(normalized_player_name);

CREATE INDEX IF NOT EXISTS idx_players_normalized_name
  ON players(normalized_player_name);

CREATE INDEX IF NOT EXISTS idx_rosters_normalized_name
  ON rosters(normalized_player_name);

CREATE TABLE IF NOT EXISTS park_factors (
  venue_id     INTEGER,
  venue_name   TEXT,
  season       INTEGER,
  park_factor  INTEGER,
  scraped_at   TEXT,
  PRIMARY KEY (venue_id, season)
);

CREATE INDEX IF NOT EXISTS idx_park_factors_season
  ON park_factors(season);
