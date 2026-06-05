import pathlib

import pytest

from src.data import database


@pytest.fixture
def temp_db_manager(tmp_path):
    db_path = tmp_path / "test.sqlite"
    schema_path = tmp_path / "schema.sql"
    schema_path.write_text(pathlib.Path("tests/assets/synthetic_schema.sql").read_text())

    previous = database._db_manager
    database._db_manager = None
    dbm = database.get_database_manager(
        db_path=db_path,
        schema_path=schema_path,
        max_connections=2,
    )
    try:
        yield dbm
    finally:
        database.close_database_connections()
        database._db_manager = previous


def test_initialize_schema_and_basic_query_helpers(temp_db_manager):
    temp_db_manager.initialize_schema(force_recreate=True)

    affected = temp_db_manager.execute_write_query(
        "INSERT INTO players (player_id, name, pos, current_team) VALUES (?, ?, ?, ?)",
        (1, "Test Player", "P", "SEA"),
    )
    assert affected == 1

    row = temp_db_manager.execute_read_query_one(
        "SELECT player_id, name FROM players WHERE player_id = ?",
        (1,),
    )
    assert row._mapping["name"] == "Test Player"

    rows = temp_db_manager.execute_read_query("SELECT name FROM players")
    assert [row._mapping["name"] for row in rows] == ["Test Player"]


def test_execute_many_write_queries(temp_db_manager):
    temp_db_manager.initialize_schema(force_recreate=True)

    affected = temp_db_manager.execute_many_write_queries(
        "INSERT INTO players (player_id, name, pos, current_team) VALUES (?, ?, ?, ?)",
        [
            (1, "First Player", "P", "SEA"),
            (2, "Second Player", "C", "SEA"),
        ],
    )

    assert affected == 2
    rows = temp_db_manager.execute_read_query(
        "SELECT player_id FROM players ORDER BY player_id"
    )
    assert [row._mapping["player_id"] for row in rows] == [1, 2]


def test_writer_connection_rolls_back_on_error(temp_db_manager):
    temp_db_manager.initialize_schema(force_recreate=True)

    with pytest.raises(database.TransactionError):
        with temp_db_manager.get_writer_connection() as conn:
            conn.exec_driver_sql(
                "INSERT INTO players (player_id, name, pos, current_team) VALUES (?, ?, ?, ?)",
                (1, "Rolled Back", "P", "SEA"),
            )
            raise RuntimeError("force rollback")

    rows = temp_db_manager.execute_read_query("SELECT * FROM players")
    assert rows == []


def test_close_database_connections_resets_global_manager(temp_db_manager):
    assert database._db_manager is temp_db_manager

    database.close_database_connections()

    assert database._db_manager is None
