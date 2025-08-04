import statsapi
import requests
import time
import json
import logging
from typing import Tuple, Union, List
from datetime import datetime, timedelta, date
from concurrent.futures import ThreadPoolExecutor
from src.config import DATES, TEAM_ABBR_MAP, TEAM_TO_TEAM_ID_STATSAPI_MAP
from data.database import get_database_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fetch_schedule')

games_requested = set()
games_received = set()

def _dh_encoder(dh_status: str, game_num: int) -> int:
    if dh_status == 'N':
        dh = 0
    elif dh_status == 'Y' or dh_status == 'S':
        if game_num == 1:
            dh = 1
        else:
            dh = 2
    else:
        raise ValueError(f"Invalid dh status: {dh_status}")
    
    return dh

def fetch_roster_for_date(team: str, date: str) -> List[Tuple[str, str, str, str]]:
    team_id = TEAM_TO_TEAM_ID_STATSAPI_MAP.get(team, None)
    if team_id == None:
        raise ValueError(f"Invalid team name: {team} team_id: {team_id}")
    
    team_data = statsapi.get('team_roster', params={'teamId': team_id, 'date': date}, force=False)
    roster = team_data['roster']
    roster_data = []

    for player in roster:
        name = player['person']['fullName']
        position = player['position']['abbreviation']
        status = player['status']['description']
        roster_data.append((name, team, position, status)) 

    return roster_data

def fetch_game_weather(gamePk: int, gameDateTime: str) -> Tuple[str, str, Union[int, None]]:
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            g = statsapi.get('game', {'gamePk': gamePk}, force=False)
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
    
    return '', '', None

def fetch_schedule():
    db_manager = get_database_manager()
    
    # Initialize schema to ensure all tables exist
    try:
        db_manager.initialize_schema(force_recreate=False)
        logger.info("Database schema initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}")
        return
    
    try:
        db_manager.execute_write_query("DELETE FROM schedule")
        logger.info("Existing schedule data cleared.")
    except Exception as e:
        logger.info(f"Schedule table does not exist yet, skipping deletion: {e}")
        
    try:
        db_manager.execute_write_query("DELETE FROM rosters")
        logger.info("Existing roster data cleared.")
    except Exception as e:
        logger.info(f"Rosters table does not exist yet, skipping deletion: {e}")
    
    total_games = 0
    scraped_at = datetime.now().isoformat()
    
    for year, (start_date, end_date) in DATES.items():
        logger.info(f"Processing {year}: {start_date} to {end_date}\n")

        end_date = end_date - timedelta(days=1)
        start_str = start_date.strftime('%m/%d/%Y')
        end_str = end_date.strftime('%m/%d/%Y')
        
        try:
            games = statsapi.schedule(start_date=start_str, end_date=end_str)
            valid_games = [game for game in games if not (game['status'] == 'Cancelled' or game['status'] == 'Postponed') and game['game_type'] == 'R']

            year_games = len(games)
            total_games += year_games
            
            logger.info(f"Found {year_games} games for {year}")
            
            batch_size = 128
            weather_failed_games = []
            roster_failed_games = []
            
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                for i in range(0, len(valid_games), batch_size):
                    batch = valid_games[i:i + batch_size]
                    
                    if i % 50 == 0:
                        logger.info(f"Processing games {i+1}-{min(i+batch_size, year_games)}/{year_games} for {year}\n")
                    weather_futures = {}
                    roster_futures = {}
                    for game in batch:

                        if game['away_name'] == 'American League All-Stars' or game['home_name'] == 'American League All-Stars':
                            logger.info(f"Skipping All-Star Game: {game['game_date']}")
                            continue
                        
                        gamePk = game['game_id']
                        status = game['status']

                        away_team = game['away_name']
                        home_team = game['home_name']
                        away_abbr = TEAM_ABBR_MAP.get(away_team, None)
                        home_abbr = TEAM_ABBR_MAP.get(home_team, None)
                        date = game['game_date']

                        if status == 'Postponed' or status == 'Cancelled':
                            logger.info(f"Skipping {status} game: {gamePk}")
                        
                        gameDateTime = game['game_datetime']
                        games_requested.add(f'{gamePk}, {gameDateTime}')
                        weather = executor.submit(fetch_game_weather, gamePk, gameDateTime)
                        
                        # Only fetch roster data if team abbreviations are valid
                        if home_abbr and away_abbr:
                            home_roster = executor.submit(fetch_roster_for_date, home_abbr, date)
                            away_roster = executor.submit(fetch_roster_for_date, away_abbr, date)
                            roster_futures[(date, home_abbr)] = home_roster
                            roster_futures[(date, away_abbr)] = away_roster
                        else:
                            logger.warning(f"Missing team abbreviation for game {gamePk}: home={home_abbr}, away={away_abbr}")
                        
                        weather_futures[gamePk] = weather
                    
                    weather_batch_data = []
                    roster_batch_data = []
                    for game in batch:
                        try:
                            if game['away_name'] == 'American League All-Stars' or game['home_name'] == 'American League All-Stars':
                                continue

                            gamePk = game['game_id']
                            game_date_str = game['game_date']
                            away_team = game['away_name']
                            home_team = game['home_name']
                            away_abbr = TEAM_ABBR_MAP.get(away_team, None)
                            home_abbr = TEAM_ABBR_MAP.get(home_team, None)
                            

                            
                            try:
                                wind, condition, temp = weather_futures[gamePk].result(timeout=30)
                            except Exception as e:
                                logger.error(f"Error getting weather for {gamePk}, will retry: {e}")
                                weather_failed_games.append((gamePk, game['game_datetime'], game))
                                wind, condition, temp = '', '', None

                            try:
                                if home_abbr and away_abbr and (game_date_str, home_abbr) in roster_futures and (game_date_str, away_abbr) in roster_futures:
                                    logger.debug(f"Fetching roster data for game {gamePk}: {away_abbr} vs {home_abbr} on {game_date_str}")
                                    away_roster_data = roster_futures[(game_date_str, away_abbr)].result(timeout=30)
                                    home_roster_data = roster_futures[(game_date_str, home_abbr)].result(timeout=30)
                                    logger.debug(f"Retrieved roster data: away={len(away_roster_data)}, home={len(home_roster_data)}")
                                else:
                                    away_roster_data = []
                                    home_roster_data = []
                                    if not (home_abbr and away_abbr):
                                        logger.warning(f"Skipping roster data for game {gamePk} due to missing team abbreviations")
                                    else:
                                        logger.warning(f"Skipping roster data for game {gamePk} - futures not found")
                            except Exception as e:
                                logger.error(f"Error getting roster data for game {gamePk}, will retry: {e}")
                                roster_failed_games.append((gamePk, game_date_str, away_abbr, home_abbr, game))
                                away_roster_data = []
                                home_roster_data = []

                            away_team = game['away_name']
                            home_team = game['home_name']
                            away_abbr = TEAM_ABBR_MAP.get(away_team, None)
                            home_abbr = TEAM_ABBR_MAP.get(home_team, None)

                            away_score = game['away_score']
                            home_score = game['home_score']

                            dh_status = game['doubleheader']
                            game_num = game['game_num']

                            dh = _dh_encoder(dh_status, game_num)
                            
                            if 'winning_team' in game and 'losing_team' in game:
                                winning_team = TEAM_ABBR_MAP.get(game['winning_team'], None)
                                losing_team = TEAM_ABBR_MAP.get(game['losing_team'], None)
                            else:
                                if away_score > home_score:
                                    winning_team = away_team
                                    losing_team = home_team
                                elif home_score > away_score:
                                    winning_team = home_team
                                    losing_team = away_team
                                else:  
                                    if not (game['status'] == 'Postponed' or game['status'] == 'Cancelled'):
                                        logger.error(f"Tie game: {gamePk}")
                                    winning_team = None
                                    losing_team = None
                                    
                            assert game['status'] != 'Postponed' or game['status'] != 'Cancelled'
                            
                            weather_batch_data.append((
                                game['game_id'],
                                game['game_date'],
                                game['game_datetime'],
                                year,
                                game['away_name'],
                                game['home_name'],
                                away_abbr,
                                home_abbr,
                                dh,
                                game['venue_name'],
                                game['venue_id'],
                                game['status'],
                                game['away_score'],
                                game['home_score'],
                                winning_team,
                                losing_team,
                                game['away_probable_pitcher'],
                                game['home_probable_pitcher'],
                                wind,
                                condition,
                                temp,
                                scraped_at
                            ))

                            # Add roster data for both teams
                            logger.debug(f"Processing roster data for game {gamePk}: away_roster={len(away_roster_data)}, home_roster={len(home_roster_data)}")
                            for player_name, team, position, status in away_roster_data:
                                roster_batch_data.append((
                                    game_date_str,
                                    year,
                                    team,
                                    player_name,
                                    position,
                                    status,
                                    scraped_at
                                ))
                            
                            for player_name, team, position, status in home_roster_data:
                                roster_batch_data.append((
                                    game_date_str,
                                    year,
                                    team,
                                    player_name,
                                    position,
                                    status,
                                    scraped_at
                                ))

                        except Exception as e:
                            status = game['status']
                            if status == 'Cancelled' or status == 'Postponed':
                                logger.info(f"Game {game['game_id']} is {game['status']} on {game['game_date']}")
                            else:
                                logger.warning(f"Error fetching game {game['game_id']} Date: {game['game_date']}")
                                logger.warning(e)
                            continue
                    
                    if weather_batch_data:
                        db_manager.execute_many_write_queries("""
                            INSERT OR REPLACE INTO schedule (
                                game_id, game_date, game_datetime, season, away_team, home_team,
                                away_team_abbr, home_team_abbr, dh, venue_name, venue_id, status,
                                away_score, home_score, winning_team, losing_team, away_probable_pitcher,
                                home_probable_pitcher, wind, condition, temp, scraped_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, weather_batch_data)
                    
                    if roster_batch_data:
                        logger.info(f"Inserting {len(roster_batch_data)} roster records into database")
                        db_manager.execute_many_write_queries("""
                            INSERT OR REPLACE INTO rosters (
                                game_date, season, team, player_name, position, status, scraped_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, roster_batch_data)
                        logger.info("Roster data inserted successfully")
                    else:
                        logger.warning("No roster data to insert for this batch")
                    
                    if weather_failed_games:
                        logger.info(f"Retrying {len(weather_failed_games)} failed weather requests...")
                        retry_updates = []
                        for gamePk, gameDateTime, game in weather_failed_games:
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
                        
                        weather_failed_games.clear()
                    
                    if roster_failed_games:
                        logger.info(f"Retrying {len(roster_failed_games)} failed roster requests...")
                        roster_retry_data = []
                        for gamePk, date, away_abbr, home_abbr, game in roster_failed_games:
                            try:
                                logger.info(f"Retrying roster fetch for game {gamePk}...")
                                # Convert string date to date object if needed for retry
                                if isinstance(date, str):
                                    date = datetime.strptime(date, '%Y-%m-%d').date()
                                
                                if away_abbr and home_abbr:
                                    away_roster_data = fetch_roster_for_date(away_abbr, date)
                                    home_roster_data = fetch_roster_for_date(home_abbr, date)
                                    
                                    # Add roster data for both teams
                                    for player_name, team, position, status in away_roster_data:
                                        roster_retry_data.append((
                                            game['game_date'],
                                            team,
                                            year,
                                            player_name,
                                            position,
                                            status,
                                            scraped_at
                                        ))
                                    
                                    for player_name, team, position, status in home_roster_data:
                                        roster_retry_data.append((
                                            game['game_date'],
                                            year,
                                            team,
                                            player_name,
                                            position,
                                            status,
                                            scraped_at
                                        ))
                                    
                                    logger.info(f"Successfully retrieved roster data on retry for game {gamePk}")
                                else:
                                    logger.warning(f"Retry failed for game {gamePk} - missing team abbreviations")
                                    
                            except Exception as retry_e:
                                logger.error(f"Roster retry also failed for game {gamePk}: {retry_e}")
                        
                        if roster_retry_data:
                            db_manager.execute_many_write_queries("""
                                INSERT OR REPLACE INTO rosters (
                                    game_date, season, team, player_name, position, status, scraped_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, roster_retry_data)
                        
                        roster_failed_games.clear()
                    
                    
                    
                    if i + batch_size < len(games):
                        time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error fetching schedule for {year}: {e}")
            continue

if __name__ == "__main__":
    fetch_schedule()
    db_manager = get_database_manager()
    