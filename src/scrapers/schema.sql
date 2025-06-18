CREATE TABLE IF NOT EXISTS players (
  player_id TEXT PRIMARY KEY,      
  name      TEXT,
  team      TEXT
);

CREATE TABLE IF NOT EXISTS batting_stats (
  player_id       TEXT, 
  game_date       TEXT,
  team            TEXT,
  games           INTEGER,
  ab              INTEGER,
  pa              INTEGER,
  ops             REAL,
  bb_k            REAL,
  wrc_plus        REAL,
  barrel_percent  REAL,
  hard_hit        REAL,
  war             REAL,
  baserunning     REAL,
  scraped_at      TEXT,
  PRIMARY KEY (player_id, game_date),
  FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS pitching_stats (
  player_id       TEXT,
  game_date       TEXT,
  team            TEXT,
  games           INTEGER,
  era             REAL,
  ip              REAL,
  k_percent       REAL,
  bb_percent      REAL,
  barrel_percent  REAL,
  hard_hit        REAL,
  war             REAL,
  siera           REAL,
  fip             REAL,
  scraped_at      TEXT,
  PRIMARY KEY (player_id, game_date),
  FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE INDEX IF NOT EXISTS idx_batting_team_date
  ON batting_stats(team, game_date);

CREATE INDEX IF NOT EXISTS idx_pitching_team_date
  ON pitching_stats(team, game_date);
