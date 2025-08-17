"""
Test configuration and fixtures for betpredictor loaders.

This module provides test fixtures for setting up isolated test databases
and helper functions for seeding test data.
"""

import sqlite3
import tempfile
import pathlib
from datetime import date, datetime
from typing import List, Tuple, Any

import pytest
import pandas as pd

from data import database


@pytest.fixture(scope="session")
def synthetic_db(tmp_path_factory):
    """
    Create a synthetic test database for all loader tests.
    
    This fixture:
    1. Creates a temporary SQLite database
    2. Initializes it with the synthetic schema
    3. Monkey-patches the global database manager
    4. Ensures all loaders use this test database
    """
    # Create temporary database and schema paths
    db_path = tmp_path_factory.mktemp("db") / "test.sqlite"
    schema_path = tmp_path_factory.mktemp("schema") / "schema.sql"
    
    # Copy synthetic schema to temp location
    synthetic_schema = pathlib.Path("tests/assets/synthetic_schema.sql").read_text()
    schema_path.write_text(synthetic_schema)
    
    # Create database manager with test database
    dbm = database.get_database_manager(
        db_path=db_path,
        schema_path=schema_path,
        max_connections=4
    )
    
    # Initialize schema
    dbm.initialize_schema(force_recreate=True)
    
    # Monkey-patch global database manager so all imports use test database
    database._db_manager = dbm
    
    yield dbm
    
    # Cleanup
    dbm.close_all_connections()


@pytest.fixture
def clean_db(synthetic_db):
    """
    Provide a clean database for each test.
    Truncates all tables before each test runs.
    """
    tables = [
        'lineup_players', 'lineups', 'batting_stats', 'pitching_stats', 
        'fielding', 'rosters', 'odds', 'schedule', 'players', 'park_factors'
    ]
    
    with synthetic_db.get_writer_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = OFF")
        for table in tables:
            cursor.execute(f"DELETE FROM {table}")
        cursor.execute("PRAGMA foreign_keys = ON")
    
    return synthetic_db


def insert_schedule_games(dbm, games: List[Tuple]) -> None:
    """
    Insert schedule/game data for testing.
    
    Args:
        dbm: Database manager instance
        games: List of tuples with game data
               (game_id, game_date, game_datetime, season, away_team, home_team, 
                status, away_score, home_score, winning_team, losing_team, dh)
    """
    
    query = """
    INSERT INTO schedule (
        game_id, game_date, game_datetime, season, away_team, home_team, 
        status, away_score, home_score, winning_team, losing_team, dh
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    dbm.execute_many_write_queries(query, games)


def insert_odds_data(dbm, odds: List[Tuple]) -> None:
    """
    Insert odds data for testing.
    
    Args:
        dbm: Database manager instance  
        odds: List of tuples with odds data
              (game_date, game_datetime, away_team, home_team, away_starter, home_starter,
               sportsbook, away_opening_odds, home_opening_odds, away_current_odds, home_current_odds, season)
    """
    
    query = """
    INSERT INTO odds (
        game_date, game_datetime, away_team, home_team, away_starter, home_starter,
        sportsbook, away_opening_odds, home_opening_odds, away_current_odds, home_current_odds, season
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    dbm.execute_many_write_queries(query, odds)


def insert_players(dbm, players: List[Tuple]) -> None:
    """
    Insert player data for testing.
    
    Args:
        dbm: Database manager instance
        players: List of tuples with player data (player_id, name, pos, current_team)
    """
    
    query = """
    INSERT INTO players (player_id, name, pos, current_team) 
    VALUES (?, ?, ?, ?)
    """
    
   
    
    dbm.execute_many_write_queries(query, players)


def insert_batting_stats(dbm, stats: List[Tuple]) -> None:
    """
    Insert batting stats for testing.
    
    Args:
        dbm: Database manager instance
        stats: List of tuples with batting data
               (player_id, game_date, team, dh, ab, pa, ops, wrc_plus, season)
    """
    query = """
    INSERT INTO batting_stats (
        player_id, game_date, team, dh, ab, pa, ops, wrc_plus, season
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    dbm.execute_many_write_queries(query, stats)


def insert_pitching_stats(dbm, stats: List[Tuple]) -> None:
    """
    Insert pitching stats for testing.
    
    Args:
        dbm: Database manager instance
        stats: List of tuples with pitching data
               (player_id, game_date, team, dh, era, ip, k_percent, season)
    """
    query = """
    INSERT INTO pitching_stats (
        player_id, game_date, team, dh, era, ip, k_percent, season
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    dbm.execute_many_write_queries(query, stats)


def insert_lineups(dbm, lineups: List[Tuple]) -> None:
    """
    Insert lineup data for testing.
    
    Args:
        dbm: Database manager instance
        lineups: List of tuples with lineup data
                (game_date, team_id, team, dh, opposing_team_id, opposing_team, team_starter_id, season)
    """
    query = """
    INSERT INTO lineups (
        game_date, team_id, team, dh, opposing_team_id, opposing_team, team_starter_id, season
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    dbm.execute_many_write_queries(query, lineups)


def insert_lineup_players(dbm, players: List[Tuple]) -> None:
    """
    Insert lineup player data for testing.
    
    Args:
        dbm: Database manager instance
        players: List of tuples with lineup player data
                (game_date, team_id, team, opposing_team_id, opposing_team, dh, player_id, position, batting_order, season)
    """
    query = """
    INSERT INTO lineup_players (
        game_date, team_id, team, opposing_team_id, opposing_team, dh, player_id, position, batting_order, season
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    dbm.execute_many_write_queries(query, players)


def insert_fielding_stats(dbm, stats: List[Tuple]) -> None:
    """
    Insert fielding stats for testing.
    
    Args:
        dbm: Database manager instance
        stats: List of tuples with fielding data (name, season, frv, total_innings)
                The normalized_player_name will be automatically generated from name
    """
    from src.utils import normalize_names
    
    query = """
    INSERT INTO fielding (name, normalized_player_name, season, frv, total_innings) 
    VALUES (?, ?, ?, ?, ?)
    """
    
    fielding_with_normalized = []
    for stat in stats:
        name, season, frv, total_innings = stat
        normalized_name = normalize_names(name)
        fielding_with_normalized.append((name, normalized_name, season, frv, total_innings))
    
    dbm.execute_many_write_queries(query, fielding_with_normalized)

def insert_rosters(dbm, roster: List[Tuple]) -> None:
    """
    Insert players on a roster
    
    Args:
        dbm: Database manager instance
        roster: List of tuples with roster date (game_date, team, player_name, position, status)
                The normalized_player_name will be automatically generated from player_name
    """
    from src.utils import normalize_names
    
    query = """
    INSERT into rosters (game_date, season, team, player_name, normalized_player_name, position, status)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    
    roster_with_normalized = []
    for entry in roster:
        game_date, season, team, player_name, position, status = entry
        normalized_name = normalize_names(player_name)
        roster_with_normalized.append((game_date, season, team, player_name, normalized_name, position, status))

    dbm.execute_many_write_queries(query, roster_with_normalized)


def insert_park_factors(dbm, park_factors: List[Tuple]) -> None:
    """
    Insert park factor data for testing.
    
    Args:
        dbm: Database manager instance
        park_factors: List of tuples with park factor data
                     (venue_id, venue_name, season, park_factor)
    """
    query = """
    INSERT INTO park_factors (venue_id, venue_name, season, park_factor) 
    VALUES (?, ?, ?, ?)
    """
    
    dbm.execute_many_write_queries(query, park_factors)


def assert_dataframe_schema(df: pd.DataFrame, expected_columns: List[str]) -> None:
    """Assert that dataframe has expected columns."""
    assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
    missing_cols = set(expected_columns) - set(df.columns)
    assert not missing_cols, f"Missing columns: {missing_cols}"


def assert_dataframe_not_empty(df: pd.DataFrame) -> None:
    """Assert that dataframe is not empty."""
    assert not df.empty, "DataFrame should not be empty"
    assert len(df) > 0, "DataFrame should have rows"


def assert_dataframe_values(df: pd.DataFrame, column: str, expected_values: List[Any]) -> None:
    """Assert that specific column contains expected values."""
    actual_values = df[column].tolist()
    assert set(actual_values) == set(expected_values), \
        f"Column {column} values {actual_values} don't match expected {expected_values}"