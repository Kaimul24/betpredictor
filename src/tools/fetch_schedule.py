import statsapi
import requests
import time
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from src.config import DATES, TEAM_ABBR_MAP
from data.database import get_database_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fetch_schedule')

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
                logger.info("Successful weather API call, but some values are null")
                logger.info(f"Game: {gamePk} Date: {gameDateTime}")
                logger.info(f"Wind: {wind}; Condition {condition}; Temp {temp}")
            return wind, condition, temp
            
        except (requests.exceptions.ConnectTimeout, 
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            
            if attempt < max_retries - 1: 
                logger.warning(f"API timeout for game {gamePk}, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"Failed to fetch weather for game {gamePk} after {max_retries} attempts: {e}")
                return '', '', None
            
        except Exception as e:
            logger.error(f"Unknown error fetching weather for game {gamePk}: {e}")
            return '', '', None

def fetch_schedule():
    failed_game_status = []
    db_manager = get_database_manager()
    
    # Clear existing schedule data
    db_manager.execute_write_query("DELETE FROM schedule")
    logger.info("Existing schedule data cleared.")
    
    total_games = 0
    scraped_at = datetime.now().isoformat()
    
    for year, (start_date, end_date) in DATES.items():
        if year == '2021':
            logger.info(f"\nProcessing {year}: {start_date} to {end_date}")
            
            start_str = start_date.strftime('%m/%d/%Y')
            end_str = end_date.strftime('%m/%d/%Y')
            
            try:
                games = statsapi.schedule(start_date=start_str, end_date=end_str)
                year_games = len(games)
                total_games += year_games
                
                logger.info(f"Found {year_games} games for {year}")
                
                batch_size = 128
                failed_games = []
                
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    for i in range(0, len(games), batch_size):
                        batch = games[i:i + batch_size]
                        
                        if i % 50 == 0:
                            logging.info(f"Processing games {i+1}-{min(i+batch_size, year_games)}/{year_games} for {year}")
                        weather_futures = {}
                        for game in batch:
                            if game['away_name'] == 'American League All-Stars' or game['home_name'] == 'American League All-Stars':
                                logging.info(f"Skipping All-Star Game: {game['game_date']}")
                                continue
                                
                            gamePk = game['game_id']
                            gameDateTime = game['game_datetime']
                            games_requested.add(f'{gamePk}, {gameDateTime}')
                            future = executor.submit(fetch_game_weather, gamePk, gameDateTime)
                            weather_futures[gamePk] = future
                        
                        batch_data = []
                        for game in batch:
                            try:
                                if game['away_name'] == 'American League All-Stars' or game['home_name'] == 'American League All-Stars':
                                    continue
                                gamePk = game['game_id']
                                
                                try:
                                    wind, condition, temp = weather_futures[gamePk].result(timeout=30)
                                except Exception as e:
                                    logging.error(f"Error getting weather for {gamePk}, will retry: {e}")
                                    failed_games.append((gamePk, game['game_datetime'], game))
                                    wind, condition, temp = '', '', None

                                away_abbr = TEAM_ABBR_MAP.get(game['away_name'], None)
                                home_abbr = TEAM_ABBR_MAP.get(game['home_name'], None)
                                
                                batch_data.append((
                                    game['game_id'],
                                    game['game_date'],
                                    game['game_datetime'],
                                    year,
                                    game['away_name'],
                                    game['home_name'],
                                    away_abbr,
                                    home_abbr,
                                    game['doubleheader'],
                                    game['game_num'],
                                    game['venue_name'],
                                    game['venue_id'],
                                    game['status'],
                                    game['away_score'],
                                    game['home_score'],
                                    game['winning_team'],
                                    game['losing_team'],
                                    game.get('away_probable_pitcher'),
                                    game.get('home_probable_pitcher'),
                                    wind,
                                    condition,
                                    temp,
                                    scraped_at
                                ))

                            except Exception as e:
                                status = game['status']
                                failed_game_status.append(status)
                                if game['game_id'] == 633481:
                                    logger.info(f"FOUND GAME 633481: CAUSED ERROR")
                                    logger.info(f"Status: {status}")
                                if status == 'Cancelled' or status == 'Postponed':
                                    logger.info(f"Game {game['game_id']} is {game['status']} on {game['game_date']}")
                                else:
                                    logger.warning(f"Error fetching game {game['game_id']} Date: {game['game_date']}")
                                    logger.warning(e)
                                continue
                        
                        if batch_data:
                            db_manager.execute_many_write_queries("""
                                INSERT OR REPLACE INTO schedule (
                                    game_id, game_date, game_datetime, season, away_team, home_team,
                                    away_team_abbr, home_team_abbr, doubleheader, game_num,
                                    venue_name, venue_id, status, away_score, home_score, 
                                    winning_team, losing_team, away_probable_pitcher, home_probable_pitcher, 
                                    wind, condition, temp, scraped_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, batch_data)
                        
                        if failed_games:
                            logger.info(f"Retrying {len(failed_games)} failed weather requests...")
                            retry_updates = []
                            for gamePk, gameDateTime, game in failed_games:
                                try:
                                    logger.info(f"Retrying weather fetch for game {gamePk}...")
                                    wind, condition, temp = fetch_game_weather(gamePk, gameDateTime)
                                    
                                    retry_updates.append((wind, condition, temp, gamePk))
                                    
                                    if wind or condition or temp is not None:
                                        logger.info(f"Successfully retrieved weather on retry for game {gamePk}")
                                    else:
                                        logger.warning(f"Retry failed for game {gamePk} - no weather data available")
                                        
                                except Exception as retry_e:
                                    logger.error(f"Retry also failed for game {gamePk}: {retry_e}")
                            
                            if retry_updates:
                                db_manager.execute_many_write_queries("""
                                    UPDATE schedule 
                                    SET wind = ?, condition = ?, temp = ?
                                    WHERE game_id = ?
                                """, retry_updates)
                            
                            failed_games.clear()
                        
                        if i + batch_size < len(games):
                            time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching schedule for {year}: {e}")
                continue
    
    print(f"\n=== SCHEDULE FETCH COMPLETE ===")
    print(f"Total games processed: {total_games}")
    
    print("\n=== DOUBLEHEADER STATISTICS ===")
    result = db_manager.execute_read_query_one("""
        SELECT 
            COUNT(*) as total_games,
            SUM(CASE WHEN doubleheader != 'N' THEN 1 ELSE 0 END) as doubleheader_games,
            COUNT(DISTINCT game_date) as unique_dates
        FROM schedule
    """)
    
    if result:
        total, dh_games, dates = result
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

    print(f'Failed Status: {set(failed_game_status)}')

    with open("weather_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    fetch_schedule()