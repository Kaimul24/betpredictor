import statsapi
import time
import logging
from typing import Tuple, List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from src.config import DATES, TEAM_ABBR_MAP, TEAM_TO_TEAM_ID_STATSAPI_MAP
from src.utils import normalize_names, normalize_datetime_string
from src.data.database import get_database_manager
from src.tools.update_table_columns import auto_update_schema_for_tool

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
        player_id = player['person']['id']
        normalized_name = normalize_names(name)
        position = player['position']['abbreviation']
        status = player['status']['description']
        roster_data.append((name, normalized_name, player_id, team, position, status)) 

    return roster_data

def fetch_schedule():
    db_manager = get_database_manager()
    
    try:
        db_manager.initialize_schema(force_recreate=False)
        logger.info("Database schema initialized.")
        
        schema_updated = auto_update_schema_for_tool("fetch_schedule")
        if not schema_updated:
            logger.warning("Schema update check failed, but continuing with schedule fetch")
            
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


        params = {
            'sportId': 1,
            'startDate': start_str,
            'endDate': end_str,
            'hydrate': 'team,weather,statusFlags,venue(timezone,location,elevation),probablePitcher'
        }
        
        try:
            data = statsapi.get('schedule', params=params, force=False)
            dates = data['dates']
            dates_data = [date['games'] for date in dates]
            all_games = [game for games in dates_data for game in games]
            valid_games = [game for game in all_games if not(game['status']['detailedState'] == 'Cancelled' or game['status']['detailedState'] == 'Postponed') and game['gameType'] == 'R']

            year_games = len(valid_games)
            total_games += year_games
            
            logger.info(f"Found {year_games} games for {year}")
            
            batch_size = 64
            roster_failed_games = []
            
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                for i in range(0, len(valid_games), batch_size):
                    batch = valid_games[i:i + batch_size]
                    
                    if i % 100 == 0:
                        logger.info(f"Processing games {i+1}-{min(i+batch_size, year_games)}/{year_games} for {year}\n")
                    roster_futures = {}

                    for game in batch:

                        away_team_data = game['teams']['away']['team']
                        home_team_data = game['teams']['home']['team']
                        away_team_name = away_team_data['name']
                        home_team_name = home_team_data['name']

                        if away_team_name == 'American League All-Stars' or home_team_name == 'American League All-Stars':
                            logger.info(f"Skipping All-Star Game: {game['officialDate']}")
                            continue

                        away_abbr = TEAM_ABBR_MAP.get(away_team_name, None)
                        home_abbr = TEAM_ABBR_MAP.get(home_team_name, None)
                        
                        gamePk = game['gamePk']
                        status = game['status']['detailedState']
                        date = game['officialDate']
                        gameDateTime = game['gameDate']

                        if status == 'Postponed' or status == 'Cancelled':
                            logger.info(f"Skipping {status} game: {gamePk}")

                        
                        if home_abbr and away_abbr:
                            home_roster = executor.submit(fetch_roster_for_date, home_abbr, date)
                            away_roster = executor.submit(fetch_roster_for_date, away_abbr, date)
                            roster_futures[(date, home_abbr)] = home_roster
                            roster_futures[(date, away_abbr)] = away_roster
                        else:
                            logger.warning(f"Missing team abbreviation for game {gamePk}: home={home_abbr}, away={away_abbr}")
                        
                    
                    schedule_batch_data = []
                    roster_batch_data = []
                    for game in batch:
                        try:
                            away_team_data = game['teams']['away']
                            home_team_data = game['teams']['home']
                            away_team_name = away_team_data['team']['name']
                            home_team_name = home_team_data['team']['name']

                            if away_team_name == 'American League All-Stars' or home_team_name == 'American League All-Stars':
                                logger.info(f"Skipping All-Star Game: {game['officialDate']}")
                                continue


                            gamePk = game['gamePk']
                            status = game['status']['detailedState']

                            away_score = away_team_data['score']
                            home_score = home_team_data['score']
                            away_winner = away_team_data['isWinner']
                            home_winner = home_team_data['isWinner']

                            away_abbr = TEAM_ABBR_MAP.get(away_team_name, None)
                            home_abbr = TEAM_ABBR_MAP.get(home_team_name, None)

                            away_probable_pitcher = away_team_data.get('probablePitcher', {}).get('fullName', None)
                            home_probable_pitcher = home_team_data.get('probablePitcher', {}).get('fullName', None)
                            away_pitcher_id = away_team_data.get('probablePitcher', {}).get('id', None)
                            home_pitcher_id = home_team_data.get('probablePitcher', {}).get('id', None)

                            if away_winner and not home_winner:
                                winning_team = away_abbr
                                losing_team = home_abbr
                            elif home_winner and not away_winner:
                                winning_team = home_abbr
                                losing_team = away_abbr
                            else:
                                raise ValueError(f"No Winner for {gamePk}\n\tAway Score: {away_abbr}, {away_score}\n\tHome Score: {home_abbr}, {home_score}")
                            
                            game_date_str = game['officialDate']
                            gameDateTime = game['gameDate']
                            day_night_game = game['dayNight']

                            weather_data = game.get('weather', {})
                            weather_condition = weather_data.get('condition', None)
                            weather_temp = weather_data.get('temp', None)
                            weather_wind = weather_data.get('wind', None)

                            venue_info = game['venue']
                            venue = venue_info['name']
                            venue_id = venue_info['id']
                            venue_elevation = venue_info['location'].get('elevation', 0)
                            venue_timezone = venue_info['timeZone']['tz']
                            venue_gametime_offset = venue_info['timeZone']['offsetAtGameTime']

                            dh_status = game['doubleHeader']
                            game_num = game['gameNumber']

                            dh = _dh_encoder(dh_status, game_num)

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
                            
                            schedule_batch_data.append((
                                gamePk,
                                game_date_str,
                                normalize_datetime_string(gameDateTime),
                                day_night_game,
                                year,
                                away_abbr,
                                home_abbr,
                                dh,
                                venue,
                                venue_id,
                                venue_elevation,
                                venue_timezone,
                                venue_gametime_offset,
                                status,
                                away_probable_pitcher,
                                home_probable_pitcher,
                                normalize_names(away_probable_pitcher),
                                normalize_names(home_probable_pitcher),
                                away_pitcher_id,
                                home_pitcher_id,
                                weather_wind,
                                weather_condition,
                                weather_temp,
                                away_score,
                                home_score,
                                winning_team,
                                losing_team,
                                scraped_at
                            ))

                            logger.debug(f"Processing roster data for game {gamePk}: away_roster={len(away_roster_data)}, home_roster={len(home_roster_data)}")
                            for player_name, normalized_player_name, player_id, team, position, status in away_roster_data:
                                roster_batch_data.append((
                                    game_date_str,
                                    year,
                                    team,
                                    player_name,
                                    normalized_player_name,
                                    player_id,
                                    position,
                                    status,
                                    scraped_at
                                ))
                            
                            for player_name, normalized_player_name, player_id, team, position, status in home_roster_data:
                                roster_batch_data.append((
                                    game_date_str,
                                    year,
                                    team,
                                    player_name,
                                    normalized_player_name,
                                    player_id,
                                    position,
                                    status,
                                    scraped_at
                                ))


                        except Exception as e:
                            status = game['status']['detailedState']
                            if status == 'Cancelled' or status == 'Postponed':
                                logger.info(f"Game {game['gamePk']} is {status} on {game['officialDate']}")
                            else:
                                logger.warning(f"Error fetching game {game['gamePk']} Date: {game['officialDate']}")
                                logger.warning(e)
                            continue
                    
                    if schedule_batch_data:
                        db_manager.execute_many_write_queries("""
                            INSERT OR REPLACE INTO schedule (
                                game_id, game_date, game_datetime, day_night_game, season, away_team, 
                                home_team, dh, venue_name, venue_id, venue_elevation, venue_timezone, 
                                venue_gametime_offset, status, away_probable_pitcher, home_probable_pitcher, 
                                away_starter_normalized, home_starter_normalized, away_pitcher_id,
                                home_pitcher_id, wind, condition, temp, away_score, home_score, winning_team,
                                losing_team,  scraped_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, schedule_batch_data)
                    
                    if roster_batch_data:
                        logger.info(f"Inserting {len(roster_batch_data)} roster records into database")
                        db_manager.execute_many_write_queries("""
                            INSERT OR REPLACE INTO rosters (
                                game_date, season, team, player_name, normalized_player_name, player_id, position, status, scraped_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, roster_batch_data)
                        logger.info("Roster data inserted successfully")
                    else:
                        logger.warning("No roster data to insert for this batch")
                    
                    if roster_failed_games:
                        logger.info(f"Retrying {len(roster_failed_games)} failed roster requests...")
                        roster_retry_data = []
                        for gamePk, date, away_abbr, home_abbr, game in roster_failed_games:
                            try:
                                logger.info(f"Retrying roster fetch for game {gamePk}...")

                                if isinstance(date, str):
                                    date = datetime.strptime(date, '%Y-%m-%d').date()
                                
                                if away_abbr and home_abbr:
                                    away_roster_data = fetch_roster_for_date(away_abbr, date)
                                    home_roster_data = fetch_roster_for_date(home_abbr, date)
                                    
                                    for player_name, normalized_name, player_id, team, position, status in away_roster_data:
                                        roster_retry_data.append((
                                            date,
                                            year,
                                            team,
                                            player_name,
                                            normalized_name,
                                            player_id, 
                                            position,
                                            status,
                                            scraped_at
                                        ))
                                    
                                    for player_name, normalized_name, player_id, team, position, status in home_roster_data:
                                        roster_retry_data.append((
                                            date,
                                            year,
                                            team,
                                            player_name,
                                            normalized_name, 
                                            player_id, 
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
                                    game_date, season, team, player_name, normalized_player_name, player_id, position, status, scraped_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, roster_retry_data)
                        
                        roster_failed_games.clear()
                    
                    
                    
                    if i + batch_size < len(valid_games):
                        time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error fetching schedule for {year}: {e}")
            continue

if __name__ == "__main__":
    fetch_schedule()
    db_manager = get_database_manager()