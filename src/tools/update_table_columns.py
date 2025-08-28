#!/usr/bin/env python3
"""
Schema Column Update Tool

This script compares the existing database schema with the schema.sql file
and adds any missing columns to existing tables. It ensures that the database
structure matches the expected schema definition.

The script:
1. Parses the schema.sql file to extract table and column definitions
2. Connects to the database and examines existing table structures
3. Identifies columns that exist in schema.sql but not in the database
4. Adds missing columns using ALTER TABLE statements
5. Handles NOT NULL columns by adding appropriate default values

Features:
- Dry-run mode to preview changes without making them
- Verbose logging for detailed operation information
- Proper handling of NOT NULL constraints with default values
- Transaction safety with rollback on errors
- Command-line interface with customizable paths
- Auto-integration with scrapers and fetch_schedule tools

Usage:
    python update_table_columns.py [--dry-run] [--verbose]
    python update_table_columns.py --db-path /path/to/db.sqlite --schema-path /path/to/schema.sql

Integration:
    The ensure_schema_updated() and auto_update_schema_for_tool() functions are automatically
    called by:
    - All Scrapy scrapers (via SqlitePipeline.open_spider())
    - fetch_schedule.py tool
    
    This ensures the database schema is always up-to-date before data insertion.
"""

import re
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

from src.config import DATABASE_PATH, SCHEMA_PATH
from src.data.database import get_database_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchemaColumnUpdater:
    """
    Tool for updating database table columns to match schema.sql file.
    """
    
    def __init__(self, db_path: Path, schema_path: Path):
        """
        Initialize the schema updater.
        
        Args:
            db_path: Path to the SQLite database
            schema_path: Path to the schema.sql file
        """
        self.db_path = db_path
        self.schema_path = schema_path
        self.db_manager = get_database_manager(db_path, schema_path)
        
    def parse_schema_file(self) -> Dict[str, Dict[str, str]]:
        """
        Parse the schema.sql file to extract table definitions and columns.
        
        Returns:
            Dictionary mapping table names to their column definitions
            Format: {table_name: {column_name: column_definition}}
        """
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        
        schema_content = self.schema_path.read_text()
        tables = {}
        
        # Pattern to match CREATE TABLE statements
        table_pattern = r'CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*?)\);'
        
        for match in re.finditer(table_pattern, schema_content, re.DOTALL | re.IGNORECASE):
            table_name = match.group(1)
            table_definition = match.group(2)
            
            # Parse column definitions
            columns = self._parse_table_columns(table_definition)
            tables[table_name] = columns
            
        logger.info(f"Parsed {len(tables)} tables from schema file")
        return tables
    
    def _parse_table_columns(self, table_definition: str) -> Dict[str, str]:
        """
        Parse column definitions from a CREATE TABLE statement.
        
        Args:
            table_definition: The content inside CREATE TABLE parentheses
            
        Returns:
            Dictionary mapping column names to their full definitions
        """
        columns = {}
        
        # Split by commas, but be careful about commas in constraints
        lines = []
        current_line = ""
        paren_depth = 0
        
        for char in table_definition:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0:
                lines.append(current_line.strip())
                current_line = ""
                continue
            current_line += char
        
        if current_line.strip():
            lines.append(current_line.strip())
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip constraints (PRIMARY KEY, FOREIGN KEY, INDEX, etc.)
            if any(keyword in line.upper() for keyword in ['PRIMARY KEY', 'FOREIGN KEY', 'INDEX', 'CONSTRAINT']):
                continue
            
            # Extract column name and definition
            parts = line.split(None, 1)
            if len(parts) >= 1:
                column_name = parts[0].strip()
                column_definition = line
                columns[column_name] = column_definition
        
        return columns
    
    def get_existing_table_columns(self) -> Dict[str, Set[str]]:
        """
        Get existing columns for all tables in the database.
        
        Returns:
            Dictionary mapping table names to sets of column names
        """
        existing_columns = {}
        
        with self.db_manager.get_reader_connection() as conn:
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get columns for each table
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = {row[1] for row in cursor.fetchall()}  # row[1] is column name
                existing_columns[table] = columns
                
        logger.info(f"Found {len(existing_columns)} existing tables in database")
        return existing_columns
    
    def find_missing_columns(self, schema_tables: Dict[str, Dict[str, str]], 
                           existing_columns: Dict[str, Set[str]]) -> Dict[str, List[str]]:
        """
        Find columns that exist in schema but not in database.
        
        Args:
            schema_tables: Table definitions from schema file
            existing_columns: Existing columns in database
            
        Returns:
            Dictionary mapping table names to lists of missing column definitions
        """
        missing_columns = defaultdict(list)
        
        for table_name, schema_cols in schema_tables.items():
            if table_name not in existing_columns:
                logger.warning(f"Table '{table_name}' exists in schema but not in database")
                continue
                
            existing_cols = existing_columns[table_name]
            
            for col_name, col_definition in schema_cols.items():
                if col_name not in existing_cols:
                    missing_columns[table_name].append(col_definition)
                    logger.info(f"Missing column in {table_name}: {col_name}")
        
        return dict(missing_columns)
    
    def _handle_not_null_column(self, col_definition: str, table_name: str, col_name: str) -> str:
        """
        Handle NOT NULL columns by adding appropriate default values.
        
        Args:
            col_definition: Original column definition
            table_name: Name of the table
            col_name: Name of the column
            
        Returns:
            Modified column definition with default value if needed
        """
        # Check if column is NOT NULL
        if 'NOT NULL' not in col_definition.upper():
            return col_definition
        
        # Remove NOT NULL for now and add default value based on data type
        definition_without_not_null = re.sub(r'\bNOT NULL\b', '', col_definition, flags=re.IGNORECASE).strip()
        
        # Determine data type and appropriate default
        col_type = definition_without_not_null.split()[1].upper() if len(definition_without_not_null.split()) > 1 else 'TEXT'
        
        if col_type.startswith('INTEGER'):
            default_value = "DEFAULT 0"
        elif col_type.startswith('REAL'):
            default_value = "DEFAULT 0.0"
        elif col_type.startswith('TEXT'):
            default_value = "DEFAULT ''"
        else:
            default_value = "DEFAULT ''"
        
        # Add default value and then NOT NULL
        modified_definition = f"{definition_without_not_null} {default_value} NOT NULL"
        
        logger.warning(f"Added default value to NOT NULL column {col_name} in {table_name}: {default_value}")
        return modified_definition
    
    def add_missing_columns(self, missing_columns: Dict[str, List[str]], dry_run: bool = False) -> None:
        """
        Add missing columns to database tables.
        
        Args:
            missing_columns: Dictionary of table names to missing column definitions
            dry_run: If True, only log what would be done without making changes
        """
        if not missing_columns:
            logger.info("No missing columns found. Database schema is up to date.")
            return
        
        if dry_run:
            logger.info("DRY RUN MODE - No changes will be made")
        
        with self.db_manager.get_writer_connection(auto_commit=False) as conn:
            cursor = conn.cursor()
            
            try:
                for table_name, column_definitions in missing_columns.items():
                    logger.info(f"Updating table '{table_name}' with {len(column_definitions)} missing columns")
                    
                    for col_definition in column_definitions:
                        # Extract column name for logging
                        col_name = col_definition.split()[0]
                        
                        # Handle NOT NULL columns by adding a default value
                        modified_definition = self._handle_not_null_column(col_definition, table_name, col_name)
                        
                        alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {modified_definition}"
                        
                        if dry_run:
                            logger.info(f"Would execute: {alter_sql}")
                        else:
                            logger.info(f"Adding column '{col_name}' to table '{table_name}'")
                            cursor.execute(alter_sql)
                
                if not dry_run:
                    conn.commit()
                    logger.info("All missing columns added successfully")
                else:
                    logger.info("Dry run completed - no changes made")
                    
            except Exception as e:
                if not dry_run:
                    conn.rollback()
                    logger.error(f"Error adding columns: {e}")
                raise
    
    def run_update(self, dry_run: bool = False, verbose: bool = False) -> None:
        """
        Run the complete schema update process.
        
        Args:
            dry_run: If True, only show what would be changed
            verbose: If True, enable verbose logging
        """
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.info("Starting schema column update process")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Schema file: {self.schema_path}")
        
        try:
            # Parse schema file
            schema_tables = self.parse_schema_file()
            
            # Get existing table columns
            existing_columns = self.get_existing_table_columns()
            
            # Find missing columns
            missing_columns = self.find_missing_columns(schema_tables, existing_columns)
            
            # Add missing columns
            self.add_missing_columns(missing_columns, dry_run)
            
            logger.info("Schema update process completed successfully")
            
        except Exception as e:
            logger.error(f"Schema update failed: {e}")
            raise


def ensure_schema_updated(db_path: Path = None, schema_path: Path = None, 
                         dry_run: bool = False, verbose: bool = False) -> bool:
    """
    Ensure the database schema is updated with any missing columns.
    
    This function is designed to be called by scrapers and other tools
    to automatically update the schema before inserting data.
    
    Args:
        db_path: Path to database file (defaults to config DATABASE_PATH)
        schema_path: Path to schema file (defaults to config SCHEMA_PATH)
        dry_run: If True, only show what would be changed
        verbose: If True, enable verbose logging
        
    Returns:
        True if update was successful or no updates needed, False if failed
    """
    # Use default paths if not provided
    if db_path is None:
        db_path = DATABASE_PATH
    if schema_path is None:
        schema_path = SCHEMA_PATH
    
    # Validate paths
    if not db_path.exists():
        logger.error(f"Database file not found: {db_path}")
        return False
    
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return False
    
    try:
        updater = SchemaColumnUpdater(db_path, schema_path)
        updater.run_update(dry_run=dry_run, verbose=verbose)
        logger.info("Schema update check completed successfully")
        return True
    except Exception as e:
        logger.error(f"Schema update failed: {e}")
        return False


def auto_update_schema_for_tool(tool_name: str = "unknown") -> bool:
    """
    Convenience function for tools to automatically update schema.
    
    This function provides a simple interface for any tool that needs
    to ensure the database schema is current before performing operations.
    
    Args:
        tool_name: Name of the calling tool (for logging purposes)
        
    Returns:
        True if update was successful or no updates needed, False if failed
    """
    logger.info(f"Auto-updating schema for tool: {tool_name}")
    return ensure_schema_updated(dry_run=False, verbose=False)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Update database table columns to match schema.sql file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making actual changes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DATABASE_PATH,
        help="Path to SQLite database file"
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=SCHEMA_PATH,
        help="Path to schema.sql file"
    )
    
    args = parser.parse_args()
    
    # Use the ensure_schema_updated function
    success = ensure_schema_updated(
        db_path=args.db_path,
        schema_path=args.schema_path,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

