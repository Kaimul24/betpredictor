"""
Database interface module for MLB stats database.

This module provides a centralized, thread-safe interface for all database
operations in the betpredictor project. It handles connection management,
transaction control, schema management, and ensures proper concurrency
with multiple readers and single writer pattern.

Classes:
    DatabaseManager: Main database interface with connection pooling and transaction management
    ReadOnlyConnection: Context manager for read-only database operations
    WriterConnection: Context manager for write operations with exclusive access
"""

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from concurrent.futures import ThreadPoolExecutor
import queue

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class ConnectionPoolError(DatabaseError):
    """Exception raised when connection pool operations fail."""
    pass


class TransactionError(DatabaseError):
    """Exception raised when transaction operations fail."""
    pass


class DatabaseManager:
    """
    Thread-safe database manager with connection pooling and transaction management.
    
    Implements a multiple-reader, single-writer pattern:
    - Multiple concurrent read operations are allowed
    - Only one write operation at a time
    - Read operations are blocked during write operations
    """
    
    def __init__(self, db_path: Union[str, Path], schema_path: Union[str, Path], 
                 max_connections: int = 10, connection_timeout: float = 30.0):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
            schema_path: Path to SQL schema file
            max_connections: Maximum number of connections in pool
            connection_timeout: Timeout for acquiring connections (seconds)
        """
        self.db_path = Path(db_path)
        self.schema_path = Path(schema_path)
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        
        self._lock = threading.RLock()
        self._reader_writer_lock = threading.RLock()
        self._active_readers = 0
        self._writer_active = False
        self._writer_waiting = False
        
        self._connection_pool: queue.Queue = queue.Queue(maxsize=max_connections)
        self._active_connections: Dict[int, sqlite3.Connection] = {}
        
        self._ensure_database_exists()
        self._initialize_connection_pool()
        
        logger.info(f"DatabaseManager initialized with {max_connections} max connections")
    
    def _ensure_database_exists(self) -> None:
        """Ensure database directory exists and create database if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.db_path.exists():
            logger.info(f"Creating new database at {self.db_path}")
            conn = sqlite3.connect(str(self.db_path))
            conn.close()
    
    def _initialize_connection_pool(self) -> None:
        """Initialize the connection pool with configured connections."""
        for _ in range(self.max_connections):
            conn = self._create_connection()
            self._connection_pool.put(conn)
    
    def _create_connection(self) -> sqlite3.Connection:
        """
        Create a new database connection with optimal settings.
        
        Returns:
            Configured SQLite connection
        """
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=self.connection_timeout
        )
        
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -64000")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 268435456") 
        
        conn.row_factory = sqlite3.Row
        
        return conn
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a connection from the pool.
        
        Returns:
            Database connection
            
        Raises:
            ConnectionPoolError: If no connection available within timeout
        """
        try:
            conn = self._connection_pool.get(timeout=self.connection_timeout)
            thread_id = threading.get_ident()
            self._active_connections[thread_id] = conn
            return conn
        except queue.Empty:
            raise ConnectionPoolError(f"No connection available within {self.connection_timeout} seconds")
    
    def _return_connection(self, conn: sqlite3.Connection) -> None:
        """
        Return a connection to the pool.
        
        Args:
            conn: Connection to return
        """
        thread_id = threading.get_ident()
        if thread_id in self._active_connections:
            del self._active_connections[thread_id]
        
        try:
            conn.rollback()
            self._connection_pool.put(conn, block=False)
        except queue.Full:
            conn.close()
    
    def _acquire_reader_lock(self) -> None:
        """Acquire reader lock (multiple readers allowed)."""
        with self._lock:
            while self._writer_active or self._writer_waiting:
                time.sleep(0.001)
            
            self._active_readers += 1
            logger.debug(f"Reader acquired lock. Active readers: {self._active_readers}")
    
    def _release_reader_lock(self) -> None:
        """Release reader lock."""
        with self._lock:
            self._active_readers -= 1
            logger.debug(f"Reader released lock. Active readers: {self._active_readers}")
    
    def _acquire_writer_lock(self) -> None:
        """Acquire writer lock (exclusive access)."""
        with self._lock:
            self._writer_waiting = True
            
            while self._active_readers > 0 or self._writer_active:
                time.sleep(0.001)
            
            self._writer_active = True
            self._writer_waiting = False
            logger.debug("Writer acquired exclusive lock")
    
    def _release_writer_lock(self) -> None:
        """Release writer lock."""
        with self._lock:
            self._writer_active = False
            logger.debug("Writer released exclusive lock")
    
    @contextmanager
    def get_reader_connection(self):
        """
        Context manager for read-only database operations.
        
        Yields:
            sqlite3.Connection: Read-only database connection
            
        Example:
            with db_manager.get_reader_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM players")
                results = cursor.fetchall()
        """
        self._acquire_reader_lock()
        conn = None
        try:
            conn = self._get_connection()
            yield conn
        finally:
            if conn:
                self._return_connection(conn)
            self._release_reader_lock()
    
    @contextmanager
    def get_writer_connection(self, auto_commit: bool = True):
        """
        Context manager for write database operations.
        
        Args:
            auto_commit: Whether to automatically commit transaction on success
            
        Yields:
            sqlite3.Connection: Database connection for writing
            
        Example:
            with db_manager.get_writer_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO players VALUES (?, ?)", (id, name))
        """
        self._acquire_writer_lock()
        conn = None
        try:
            conn = self._get_connection()
            if auto_commit:
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            if auto_commit:
                conn.commit()
        except Exception as e:
            if conn and auto_commit:
                conn.rollback()
            raise TransactionError(f"Transaction failed: {e}") from e
        finally:
            if conn:
                self._return_connection(conn)
            self._release_writer_lock()
    
    def execute_read_query(self, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """
        Execute a read-only query and return all results.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of query results
        """
        with self.get_reader_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_read_query_one(self, query: str, params: Tuple = ()) -> Optional[sqlite3.Row]:
        """
        Execute a read-only query and return first result.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            First query result or None
        """
        with self.get_reader_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()
    
    def execute_write_query(self, query: str, params: Tuple = ()) -> int:
        """
        Execute a write query and return affected rows.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        with self.get_writer_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.rowcount
    
    def execute_many_write_queries(self, query: str, params_list: List[Tuple]) -> int:
        """
        Execute multiple write queries in a single transaction.
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
            
        Returns:
            Number of affected rows
        """
        with self.get_writer_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            return cursor.rowcount
    
    def initialize_schema(self, force_recreate: bool = False) -> None:
        """
        Initialize or recreate database schema.
        
        Args:
            force_recreate: Whether to drop existing tables first
        """
        if not self.schema_path.exists():
            raise DatabaseError(f"Schema file not found: {self.schema_path}")
        
        schema_sql = self.schema_path.read_text()
        
        with self.get_writer_connection(auto_commit=False) as conn:
            try:
                if force_recreate:
                    # Get all table names and drop them
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    for table in tables:
                        cursor.execute(f"DROP TABLE IF EXISTS {table}")
                    
                    logger.info("Dropped existing tables")
                
                
                # Execute schema
                conn.executescript(schema_sql)
                conn.commit()
                logger.info("Schema initialized successfully")
                
            except Exception as e:
                conn.rollback()
                raise DatabaseError(f"Failed to initialize schema: {e}") from e
    
    # def _migrate_existing_schema(self, conn: sqlite3.Connection, tables_to_update: List[str]) -> None:
    #     """
    #     Migrate existing schema to add missing columns.
        
    #     Args:
    #         conn: Database connection
    #     """
    #     cursor = conn.cursor()
        
    #     cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    #     existing_tables = [row[0] for row in cursor.fetchall()]
        
    #     # Tables that should have a season column
    #     tables_to_update = [
    #         'schedule', 'odds', 'batting_stats', 'pitching_stats', 
    #         'lineups', 'lineup_players', 'fielding'
    #     ]
        
    #     for table in tables_to_update:
    #         if table in existing_tables:
    #             # Check if season column exists
    #             cursor.execute(f"PRAGMA table_info({table})")
    #             columns = [col[1] for col in cursor.fetchall()]
                
    #             if 'season' not in columns:
    #                 logger.info(f"Adding season column to {table}")
    #                 try:
    #                     # Add the season column
    #                     cursor.execute(f"ALTER TABLE {table} ADD COLUMN season INTEGER")
                        
    #                     # Populate season from game_date if that column exists
    #                     if 'game_date' in columns:
    #                         cursor.execute(f"""
    #                             UPDATE {table} 
    #                             SET season = CAST(substr(game_date, 1, 4) AS INTEGER) 
    #                             WHERE season IS NULL
    #                         """)
    #                         logger.info(f"Populated season column for {table}")
    #                 except Exception as e:
    #                     logger.warning(f"Failed to migrate {table}: {e}")
    #                     # If the table doesn't exist in the new schema, it's okay to skip
    #                     pass
    
    def create_indexes(self, index_queries: list[str]) -> None:
        """Create additional indexes for better query performance."""
        with self.get_writer_connection() as conn:
            cursor = conn.cursor()
            for query in index_queries:
                cursor.execute(query)
            logger.info("Additional indexes created")
    
    def vacuum_database(self) -> None:
        """Vacuum database to reclaim space and optimize performance."""
        with self.get_writer_connection(auto_commit=False) as conn:
            conn.execute("VACUUM")
            logger.info("Database vacuumed successfully")
    
    def analyze_database(self) -> None:
        """Update database statistics for query optimization."""
        with self.get_writer_connection() as conn:
            conn.execute("ANALYZE")
            logger.info("Database analyzed successfully")
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information and statistics.
        
        Returns:
            Dictionary with database information
        """
        with self.get_reader_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            db_size = page_count * page_size
            
            table_counts = {}
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                table_counts[table] = cursor.fetchone()[0]
            
            return {
                "database_path": str(self.db_path),
                "database_size_bytes": db_size,
                "database_size_mb": round(db_size / (1024 * 1024), 2),
                "tables": tables,
                "table_row_counts": table_counts,
                "active_readers": self._active_readers,
                "writer_active": self._writer_active,
                "connection_pool_size": self._connection_pool.qsize(),
                "active_connections": len(self._active_connections)
            }
    
    def check_connection_health(self) -> bool:
        """
        Check if database connections are healthy.
        
        Returns:
            True if connections are healthy, False otherwise
        """
        try:
            with self.get_reader_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result[0] == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def close_all_connections(self) -> None:
        """Close all connections in the pool. Call this when shutting down."""
        logger.info("Closing all database connections")
        
        for conn in self._active_connections.values():
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing active connection: {e}")
        self._active_connections.clear()
        
        while not self._connection_pool.empty():
            try:
                conn = self._connection_pool.get_nowait()
                conn.close()
            except (queue.Empty, Exception) as e:
                if not isinstance(e, queue.Empty):
                    logger.warning(f"Error closing pooled connection: {e}")
                break
        
        logger.info("All database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager(db_path: Union[str, Path] = None, 
                        schema_path: Union[str, Path] = None,
                        max_connections: int = 10) -> DatabaseManager:
    """
    Get or create the global database manager instance.
    
    Args:
        db_path: Path to database file (only used on first call)
        schema_path: Path to schema file (only used on first call)
        max_connections: Maximum connections in pool (only used on first call)
        
    Returns:
        DatabaseManager instance
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
    """
    Initialize the database schema.
    
    Args:
        force_recreate: Whether to recreate all tables
    """
    db_manager = get_database_manager()
    db_manager.initialize_schema(force_recreate)


def close_database_connections() -> None:
    """Close all database connections."""
    global _db_manager
    if _db_manager:
        _db_manager.close_all_connections()
        _db_manager = None


@contextmanager
def get_db_connection(readonly: bool = True):
    """
    Convenience context manager for database connections.
    
    Args:
        readonly: Whether connection is read-only
        
    Yields:
        Database connection
    """
    db_manager = get_database_manager()
    if readonly:
        with db_manager.get_reader_connection() as conn:
            yield conn
    else:
        with db_manager.get_writer_connection() as conn:
            yield conn


def execute_query(query: str, params: Tuple = (), readonly: bool = True) -> Union[List[sqlite3.Row], int]:
    """
    Execute a database query.
    
    Args:
        query: SQL query
        params: Query parameters
        readonly: Whether query is read-only
        
    Returns:
        Query results (list) for read queries, affected rows (int) for write queries
    """
    db_manager = get_database_manager()
    if readonly:
        return db_manager.execute_read_query(query, params)
    else:
        return db_manager.execute_write_query(query, params)


def execute_query_one(query: str, params: Tuple = ()) -> Optional[sqlite3.Row]:
    """
    Execute a query and return first result.
    
    Args:
        query: SQL query
        params: Query parameters
        
    Returns:
        First result or None
    """
    db_manager = get_database_manager()
    return db_manager.execute_read_query_one(query, params)
