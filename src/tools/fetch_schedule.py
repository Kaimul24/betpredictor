import statsapi
from datetime import datetime
from ..config import DATES, TEAM_ABBR_MAP, connect_database

def fetch_schedule():
    """Fetch MLB schedule for all dates in DATES config."""
    conn, cursor = connect_database()
    
    total_games = 0
    scraped_at = datetime.now().isoformat()
    
    print("Fetching MLB schedule data")
    
    for year, (start_date, end_date) in DATES.items():
        print(f"\nProcessing {year}: {start_date} to {end_date}")
        
        start_str = start_date.strftime('%m/%d/%Y')
        end_str = end_date.strftime('%m/%d/%Y')
        
        try:
            games = statsapi.schedule(start_date=start_str, end_date=end_str)
            year_games = len(games)
            total_games += year_games
            
            print(f"Found {year_games} games for {year}")
            
            for game in games:
                
                if game['away_name'] == 'American League All-Stars' or game['home_name'] == 'American League All-Stars':
                    print(f"SKIPPING ALL STAR GAME {game['game_date']}")
                    continue

                away_abbr = TEAM_ABBR_MAP.get(game['away_name'], game['away_name'])
                home_abbr = TEAM_ABBR_MAP.get(game['home_name'], game['home_name'])
                
                cursor.execute("""
                    INSERT OR REPLACE INTO schedule (
                        game_id, game_date, away_team, home_team,
                        away_team_abbr, home_team_abbr, doubleheader, game_num,
                        venue_name, venue_id, status,
                        away_probable_pitcher, home_probable_pitcher, scraped_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game['game_id'],
                    game['game_date'],
                    game['away_name'],
                    game['home_name'],
                    away_abbr,
                    home_abbr,
                    game['doubleheader'],
                    game['game_num'],
                    game['venue_name'],
                    game['venue_id'],
                    game['status'],
                    game.get('away_probable_pitcher'),
                    game.get('home_probable_pitcher'),
                    scraped_at
                ))
            
            conn.commit()
            
        except Exception as e:
            print(f"Error fetching schedule for {year}: {e}")
            continue
    
    print(f"\n=== SCHEDULE FETCH COMPLETE ===")
    print(f"Total games processed: {total_games}")
    
    print("\n=== DOUBLEHEADER STATISTICS ===")
    cursor.execute("""
        SELECT 
            COUNT(*) as total_games,
            SUM(CASE WHEN doubleheader != 'N' THEN 1 ELSE 0 END) as doubleheader_games,
            COUNT(DISTINCT game_date) as unique_dates
        FROM schedule
    """)
    
    total, dh_games, dates = cursor.fetchone()
    print(f"Total games: {total}")
    print(f"Doubleheader games: {dh_games}")
    print(f"Unique dates: {dates}")

if __name__ == "__main__":
    fetch_schedule()

