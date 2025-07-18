import statsapi
import requests
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from config import DATES, TEAM_ABBR_MAP, connect_database

games_requested = set()
games_received = set()

def fetch_game_weather(gamePk, gameDateTime):
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            g = statsapi.get('game', {'gamePk': gamePk}, force=True)
            weather = g['gameData']['weather']
            games_received.add(f'{gamePk}, {gameDateTime}')
            wind = weather.get('wind', '')
            condition = weather.get('condition', '')
            temp = weather.get('temp', None)

            if wind == '' or condition =='' or temp == None:
                print("Successful weather API call, but some values are null")
                print(f"Game: {gamePk} Date: {gameDateTime}")
                print(f"Wind: {wind}; Condition {condition}; Temp {temp}")
                print()
            return wind, condition, temp
            
        except (requests.exceptions.ConnectTimeout, 
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            
            if attempt < max_retries - 1: 
                print(f"API timeout for game {gamePk}, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Failed to fetch weather for game {gamePk} after {max_retries} attempts: {e}")
                return '', '', None
        except Exception as e:
            print(f"Unknown error fetching weather for game {gamePk}: {e}")
            return '', '', None

def fetch_schedule():
    conn, cursor = connect_database()
    conn.commit()
    
    total_games = 0
    scraped_at = datetime.now().isoformat()
    
    for year, (start_date, end_date) in DATES.items():
        print(f"\nProcessing {year}: {start_date} to {end_date}")
        
        start_str = start_date.strftime('%m/%d/%Y')
        end_str = end_date.strftime('%m/%d/%Y')
        
        try:
            games = statsapi.schedule(start_date=start_str, end_date=end_str)
            year_games = len(games)
            total_games += year_games
            
            print(f"Found {year_games} games for {year}")
            
            batch_size = 256
            failed_games = []
            
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                for i in range(0, len(games), batch_size):
                    batch = games[i:i + batch_size]
                    
                    if i % 50 == 0:
                        print(f"Processing games {i+1}-{min(i+batch_size, year_games)}/{year_games} for {year}")
                        print()
                    weather_futures = {}
                    for game in batch:
                        if game['away_name'] == 'American League All-Stars' or game['home_name'] == 'American League All-Stars':
                            print(f"SKIPPING ALL STAR GAME {game['game_date']}")
                            print()
                            continue
                            
                        gamePk = game['game_id']
                        gameDateTime = game['game_datetime']
                        games_requested.add(f'{gamePk}, {gameDateTime}')
                        future = executor.submit(fetch_game_weather, gamePk, gameDateTime)
                        weather_futures[gamePk] = future
                    
                    for game in batch:
                        try:
                            gamePk = game['game_id']
                            
                            try:
                                wind, condition, temp = weather_futures[gamePk].result(timeout=30)
                            except Exception as e:
                                print(f"Error getting weather for {gamePk}: {e}")
                                failed_games.append((gamePk, game['game_datetime'], game))
                                wind, condition, temp = '', '', None

                            away_abbr = TEAM_ABBR_MAP.get(game['away_name'], game['away_name'])
                            home_abbr = TEAM_ABBR_MAP.get(game['home_name'], game['home_name'])
                            
                            cursor.execute("""
                                INSERT OR REPLACE INTO schedule (
                                    game_id, game_date, game_datetime, away_team, home_team,
                                    away_team_abbr, home_team_abbr, doubleheader, game_num,
                                    venue_name, venue_id, status,
                                    away_probable_pitcher, home_probable_pitcher, 
                                    wind, condition, temp, scraped_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                game['game_id'],
                                game['game_date'],
                                game['game_datetime'],
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
                                wind,
                                condition,
                                temp,
                                scraped_at
                            ))
                        except Exception as e:
                            print(f"Error processing game {game.get('game_id', 'unknown')}: {e}")
                            continue
                    
                    conn.commit()
                    
                    if failed_games:
                        print(f"Retrying {len(failed_games)} failed weather requests...")
                        print()
                        for gamePk, gameDateTime, game in failed_games:
                            try:
                                print(f"Retrying weather fetch for game {gamePk}...")
                                wind, condition, temp = fetch_game_weather(gamePk, gameDateTime)
                                
                                away_abbr = TEAM_ABBR_MAP.get(game['away_name'], game['away_name'])
                                home_abbr = TEAM_ABBR_MAP.get(game['home_name'], game['home_name'])
                                
                                cursor.execute("""
                                    UPDATE schedule 
                                    SET wind = ?, condition = ?, temp = ?
                                    WHERE game_id = ?
                                """, (wind, condition, temp, gamePk))
                                
                                if wind or condition or temp is not None:
                                    print(f"Successfully retrieved weather on retry for game {gamePk}")
                                else:
                                    print(f"Retry failed for game {gamePk} - no weather data available")
                                    
                            except Exception as retry_e:
                                print(f"Retry also failed for game {gamePk}: {retry_e}")
                        
                        conn.commit()
                        failed_games.clear()
                    
                    if i + batch_size < len(games):
                        time.sleep(0.5)
            
        except Exception as e:
            print(f"Error fetching schedule for {year}: {e}")
            print()
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

    requested = sorted(games_requested)
    received = sorted(games_received)
    missing = sorted(set(requested) - set(received))

    manifest = {
            "requested": requested,
            "received":   received,
            "missing":   missing,
    }

    print(f'Requested: {len(requested)}')
    print(f'Received: {len(received)}')
    print(f'Missing: {len(missing)}')

    with open("weather_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    fetch_schedule()

