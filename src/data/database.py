"""
SQLAlchemy-backed database interface for the MLB stats database.

This module keeps the existing DatabaseManager API used throughout the project
while delegating SQLite connection pooling and transaction handling to
SQLAlchemy Core.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Connection, Engine, Row
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations."""


class ConnectionPoolError(DatabaseError):
    """Exception raised when connection pool operations fail."""


class TransactionError(DatabaseError):
    """Exception raised when transaction operations fail."""


class DatabaseManager:
    """
    Database manager backed by a SQLAlchemy Core Engine.

    Public methods intentionally mirror the previous sqlite3 wrapper so callers
    can continue using raw SQL strings and positional "?" parameters.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        schema_path: Union[str, Path],
        max_connections: int = 10,
        connection_timeout: float = 30.0,
    ):
        self.db_path = Path(db_path)
        self.schema_path = Path(schema_path)
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout

        self._ensure_database_directory()
        self.engine = self._create_engine()

        logger.info("DatabaseManager initialized with SQLAlchemy engine")

    def _ensure_database_directory(self) -> None:
        """Ensure the SQLite database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_engine(self) -> Engine:
        """Create a SQLite SQLAlchemy engine with the project's PRAGMAs."""
        engine = create_engine(
            f"sqlite:///{self.db_path}",
            future=True,
            poolclass=QueuePool,
            pool_size=self.max_connections,
            max_overflow=0,
            pool_timeout=self.connection_timeout,
            connect_args={
                "check_same_thread": False,
                "timeout": self.connection_timeout,
            },
        )

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys = ON")
                cursor.execute("PRAGMA journal_mode = WAL")
                cursor.execute("PRAGMA synchronous = NORMAL")
                cursor.execute("PRAGMA cache_size = -64000")
                cursor.execute("PRAGMA temp_store = MEMORY")
                cursor.execute("PRAGMA mmap_size = 268435456")
            finally:
                cursor.close()

        return engine

    @staticmethod
    def _normalize_params(params: Tuple | List | None) -> Tuple:
        if params is None:
            return ()
        if isinstance(params, tuple):
            return params
        return tuple(params)

    @contextmanager
    def get_reader_connection(self):
        """
        Context manager for read database operations.

        Yields:
            sqlalchemy.engine.Connection
        """
        try:
            with self.engine.connect() as conn:
                yield conn
        except SQLAlchemyTimeoutError as e:
            raise ConnectionPoolError(
                f"No connection available within {self.connection_timeout} seconds"
            ) from e

    @contextmanager
    def get_writer_connection(self, auto_commit: bool = True):
        """
        Context manager for write database operations.

        Args:
            auto_commit: Whether to commit the transaction on success.

        Yields:
            sqlalchemy.engine.Connection
        """
        try:
            if auto_commit:
                with self.engine.begin() as conn:
                    yield conn
            else:
                with self.engine.connect() as conn:
                    transaction = conn.begin()
                    try:
                        yield conn
                    except Exception:
                        transaction.rollback()
                        raise
                    else:
                        transaction.commit()
        except SQLAlchemyTimeoutError as e:
            raise ConnectionPoolError(
                f"No connection available within {self.connection_timeout} seconds"
            ) from e
        except Exception as e:
            raise TransactionError(f"Transaction failed: {e}") from e

    def execute_read_query(self, query: str, params: Tuple = ()) -> List[Row]:
        """Execute a read query and return all rows."""
        with self.get_reader_connection() as conn:
            result = conn.exec_driver_sql(query, self._normalize_params(params))
            return result.fetchall()

    def execute_read_query_one(self, query: str, params: Tuple = ()) -> Optional[Row]:
        """Execute a read query and return the first row, if any."""
        with self.get_reader_connection() as conn:
            result = conn.exec_driver_sql(query, self._normalize_params(params))
            return result.fetchone()

    def execute_write_query(self, query: str, params: Tuple = ()) -> int:
        """Execute a write query and return the affected row count."""
        with self.get_writer_connection() as conn:
            result = conn.exec_driver_sql(query, self._normalize_params(params))
            return result.rowcount

    def execute_many_write_queries(self, query: str, params_list: List[Tuple]) -> int:
        """Execute an executemany write query in one transaction."""
        if not params_list:
            return 0

        with self.get_writer_connection() as conn:
            result = conn.exec_driver_sql(query, params_list)
            return result.rowcount

    def initialize_schema(self, force_recreate: bool = False) -> None:
        """
        Initialize or recreate the database schema from schema.sql.

        Args:
            force_recreate: Whether to drop existing tables first.
        """
        if not self.schema_path.exists():
            raise DatabaseError(f"Schema file not found: {self.schema_path}")

        schema_sql = self.schema_path.read_text()

        raw_conn = self.engine.raw_connection()
        try:
            cursor = raw_conn.cursor()
            try:
                if force_recreate:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    for table in tables:
                        cursor.execute(f"DROP TABLE IF EXISTS {table}")
                    logger.info("Dropped existing tables")

                raw_conn.executescript(schema_sql)
                raw_conn.commit()
                logger.info("Schema initialized successfully")
            finally:
                cursor.close()
        except Exception as e:
            raw_conn.rollback()
            raise DatabaseError(f"Failed to initialize schema: {e}") from e
        finally:
            raw_conn.close()

    def create_indexes(self, index_queries: list[str]) -> None:
        """Create additional indexes for better query performance."""
        with self.get_writer_connection() as conn:
            for query in index_queries:
                conn.exec_driver_sql(query)
            logger.info("Additional indexes created")

    def vacuum_database(self) -> None:
        """Vacuum database to reclaim space and optimize performance."""
        with self.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.exec_driver_sql("VACUUM")
            logger.info("Database vacuumed successfully")

    def analyze_database(self) -> None:
        """Update database statistics for query optimization."""
        with self.get_writer_connection() as conn:
            conn.exec_driver_sql("ANALYZE")
            logger.info("Database analyzed successfully")

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and table row counts."""
        with self.get_reader_connection() as conn:
            result = conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in result.fetchall()]

            page_count = conn.exec_driver_sql("PRAGMA page_count").fetchone()[0]
            page_size = conn.exec_driver_sql("PRAGMA page_size").fetchone()[0]
            db_size = page_count * page_size

            table_counts = {}
            for table in tables:
                table_counts[table] = conn.exec_driver_sql(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]

            pool = self.engine.pool
            return {
                "database_path": str(self.db_path),
                "database_size_bytes": db_size,
                "database_size_mb": round(db_size / (1024 * 1024), 2),
                "tables": tables,
                "table_row_counts": table_counts,
                "pool_size": pool.size() if hasattr(pool, "size") else None,
                "checked_in_connections": (
                    pool.checkedin() if hasattr(pool, "checkedin") else None
                ),
                "checked_out_connections": (
                    pool.checkedout() if hasattr(pool, "checkedout") else None
                ),
            }

    def check_connection_health(self) -> bool:
        """Return True if the engine can execute a simple query."""
        try:
            with self.get_reader_connection() as conn:
                result = conn.exec_driver_sql("SELECT 1").fetchone()
                return result[0] == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def close_all_connections(self) -> None:
        """Dispose of all pooled SQLAlchemy connections."""
        logger.info("Disposing database engine")
        self.engine.dispose()


_db_manager: Optional[DatabaseManager] = None


def get_database_manager(
    db_path: Union[str, Path] = None,
    schema_path: Union[str, Path] = None,
    max_connections: int = 10,
) -> DatabaseManager:
    """
    Get or create the global database manager instance.

    Args:
        db_path: Path to database file, only used on first call.
        schema_path: Path to SQL schema file, only used on first call.
        max_connections: Engine pool size, only used on first call.
    """
    global _db_manager

    if _db_manager is None:
        if db_path is None or schema_path is None:
            try:
                from src.config import DATABASE_PATH, SCHEMA_PATH

                db_path = db_path or DATABASE_PATH
                schema_path = schema_path or SCHEMA_PATH
            except ImportError:
                raise DatabaseError("Database paths must be provided on first call")

        _db_manager = DatabaseManager(db_path, schema_path, max_connections)

    return _db_manager


def initialize_database(force_recreate: bool = False) -> None:
    """Initialize the database schema."""
    db_manager = get_database_manager()
    db_manager.initialize_schema(force_recreate)


def close_database_connections() -> None:
    """Close all database connections and reset the global manager."""
    global _db_manager
    if _db_manager:
        _db_manager.close_all_connections()
        _db_manager = None


@contextmanager
def get_db_connection(readonly: bool = True):
    """
    Convenience context manager for database connections.

    Yields a SQLAlchemy Connection.
    """
    db_manager = get_database_manager()
    if readonly:
        with db_manager.get_reader_connection() as conn:
            yield conn
    else:
        with db_manager.get_writer_connection() as conn:
            yield conn


def execute_query(query: str, params: Tuple = (), readonly: bool = True) -> Union[List[Row], int]:
    """
    Execute a database query.

    Returns query rows for read queries and affected row count for writes.
    """
    db_manager = get_database_manager()
    if readonly:
        return db_manager.execute_read_query(query, params)
    return db_manager.execute_write_query(query, params)


def execute_query_one(query: str, params: Tuple = ()) -> Optional[Row]:
    """Execute a query and return the first row."""
    db_manager = get_database_manager()
    return db_manager.execute_read_query_one(query, params)
