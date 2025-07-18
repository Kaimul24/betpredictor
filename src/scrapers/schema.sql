CREATE TABLE IF NOT EXISTS players (
  player_id TEXT PRIMARY KEY,      
  name      TEXT,
  current_team TEXT,
  last_updated TEXT
);

CREATE TABLE IF NOT EXISTS schedule (
  game_id       TEXT PRIMARY KEY,
  game_date     TEXT NOT NULL,
  game_datetime TEXT NOT NULL,
  away_team     TEXT NOT NULL,
  home_team     TEXT NOT NULL,
  away_team_abbr TEXT,
  home_team_abbr TEXT,
  doubleheader  TEXT,
  game_num      INTEGER,
  venue_name    TEXT,
  venue_id      INTEGER,
  status        TEXT,
  away_probable_pitcher TEXT,
  home_probable_pitcher TEXT,
  wind          TEXT,
  condition     TEXT,
  temp          INTEGER,
  scraped_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_schedule_date
  ON schedule(game_date);

CREATE INDEX IF NOT EXISTS idx_schedule_teams
  ON schedule(away_team, home_team);

CREATE TABLE IF NOT EXISTS odds (
  game_date     TEXT,
  away_team     TEXT,
  home_team     TEXT,
  away_starter  TEXT,
  home_starter  TEXT,
  away_score    INTEGER,
  home_score    INTEGER,
  winner        TEXT,
  sportsbook    TEXT,
  away_odds     REAL,
  home_odds     REAL,
  PRIMARY KEY (game_date, away_team, home_team, sportsbook)
);

CREATE INDEX IF NOT EXISTS idx_odds_date
  ON odds(game_date);

CREATE TABLE IF NOT EXISTS batting_stats (
  player_id       TEXT, 
  game_date       TEXT,
  team            TEXT,
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
  baserunning     REAL,
  scraped_at      TEXT,
  PRIMARY KEY (player_id, game_date, dh),
  FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS pitching_stats (
  player_id       TEXT,
  game_date       TEXT,
  team            TEXT,
  dh              INTEGER,
  era             REAL,
  babip           REAL,
  ip              REAL,
  k_percent       REAL,
  bb_percent      REAL,
  barrel_percent  REAL,
  hard_hit        REAL,
  siera           REAL,
  fip             REAL,
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

CREATE TABLE IF NOT EXISTS lineups (
  game_date     TEXT NOT NULL,
  team_id       INTEGER NOT NULL,
  dh            INTEGER NOT NULL,
  opposing_team_id INTEGER NOT NULL,
  team_starter_id TEXT,
  opposing_starter_id TEXT,
  scraped_at    TEXT,
  PRIMARY KEY (game_date, team_id, dh)
);

CREATE TABLE IF NOT EXISTS lineup_players (
  game_date     TEXT NOT NULL,
  team_id       INTEGER NOT NULL,
  dh            INTEGER NOT NULL,
  player_id     TEXT NOT NULL,
  position      TEXT NOT NULL,
  batting_order INTEGER,
  scraped_at    TEXT,
  PRIMARY KEY (game_date, team_id, dh, player_id)
);

CREATE INDEX IF NOT EXISTS idx_lineups_date
  ON lineups(game_date);

CREATE INDEX IF NOT EXISTS idx_lineup_players_date
  ON lineup_players(game_date);

CREATE INDEX IF NOT EXISTS idx_lineup_players_player
  ON lineup_players(player_id, game_date);

CREATE INDEX IF NOT EXISTS idx_lineup_players_position
  ON lineup_players(position, game_date);

CREATE INDEX IF NOT EXISTS idx_lineup_players_batting_order
  ON lineup_players(batting_order, game_date);
