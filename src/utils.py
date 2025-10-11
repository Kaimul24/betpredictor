from datetime import date, timedelta, datetime
import logging
import sys
import unicodedata
import re
from pathlib import Path
from typing import List, Optional, Union


def setup_logging(
    logger_name: str,
    log_file: Optional[Union[Path, str]] = None,
    args=None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    propagate: bool = False,
    base_logger: Optional[logging.Logger] = None,
):
    """
    Configure and return a logger with consistent console and file handlers.

    Args:
        logger_name: Name of the logger to configure.
        log_file: Default path for the log file when file logging is enabled.
        args: Optional argparse namespace with log-related arguments.
        console_level: Logging level for the console stream handler.
        file_level: Logging level for the optional file handler.
        propagate: Whether to propagate log records to parent loggers.
        base_logger: Existing logger whose handlers should be reused.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)

    if base_logger is logger:
        base_logger = None

    logger.setLevel(logging.DEBUG)
    logger.propagate = propagate
    logger.handlers.clear()

    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    handlers: List[logging.Handler] = []
    if base_logger and base_logger.handlers:
        handlers.extend(base_logger.handlers)

    if not handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(console_level)
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    log_messages: List[str] = []

    if not base_logger and log_file and bool(args and getattr(args, "log", False)):
        log_path = Path(getattr(args, "log_file", "") or log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if args and getattr(args, "clear_log", False):
            log_path.write_text("")
            log_messages.append(f"Cleared log file: {log_path}")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
        log_messages.append(f"Logging to file: {log_path}")

    for handler in handlers:
        logger.addHandler(handler)

    for message in log_messages:
        logger.info(message)

    return logger


def daterange(start_date: date, end_date: date):
    days = int((end_date - start_date).days)
    for n in range(days):
        yield start_date + timedelta(n)


def normalize_names(name: str) -> str:
    """
    Normalize player names for consistent identification across data sources.
    
    This function:
    1. Removes accent marks and diacritics
    2. Removes punctuation (periods, apostrophes, hyphens, etc.)
    3. Converts to lowercase
    4. Handles common name format variations (Last, First -> First Last)
    5. Removes extra whitespace
    6. Handles special cases like Jr., Sr., III, etc.
    
    Args:
        name (str): The original player name
        
    Returns:
        str: Normalized player name
        
    Examples:
        normalize_names("José Altuve") -> "jose altuve"
        normalize_names("O'Neill, Tyler") -> "tyler oneill"
        normalize_names("Guerrero Jr., Vladimir") -> "vladimir guerrero jr"
        normalize_names("de la Cruz, Elly") -> "elly de la cruz"
    """
    if not name or not isinstance(name, str):
        return ""
    
    # Remove accent marks and convert to ASCII
    normalized = unicodedata.normalize('NFD', name)
    ascii_name = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    
    # Convert to lowercase
    ascii_name = ascii_name.lower()
    
    # Handle "Last, First" format -> "First Last"
    if ',' in ascii_name:
        parts = [part.strip() for part in ascii_name.split(',')]
        if len(parts) == 2:
            ascii_name = f"{parts[1]} {parts[0]}"
    
    # Remove punctuation but preserve spaces and handle special cases
    # Keep letters, numbers, spaces, and handle suffixes
    ascii_name = re.sub(r"[^\w\s]", "", ascii_name)  # Remove punctuation
    
    # Handle multiple spaces and strip
    ascii_name = re.sub(r'\s+', ' ', ascii_name).strip()
    
    # Handle common suffix variations
    suffixes = ['jr', 'sr', 'ii', 'iii', 'iv', 'v']
    name_parts = ascii_name.split()
    
    # Move suffix to end if it's not already there
    if len(name_parts) > 2:
        for suffix in suffixes:
            if suffix in name_parts and name_parts[-1] != suffix:
                name_parts.remove(suffix)
                name_parts.append(suffix)
                break
    
    return ' '.join(name_parts)


def normalize_datetime_string(dt_string: str) -> str:
    """
    Normalize datetime strings to a common format for matching between tables.
    
    This function handles various datetime string formats and converts them to
    a standardized ISO format (YYYY-MM-DDTHH:MM:SS).
    
    Args:
        dt_string (str): The datetime string to normalize
        
    Returns:
        str: Normalized datetime string in format 'YYYY-MM-DDTHH:MM:SS'
        
    Examples:
        normalize_datetime_string("2021-04-13T18:10:00+00:00") -> "2021-04-13T18:10:00"
        normalize_datetime_string("2021-04-13T18:10:00Z") -> "2021-04-13T18:10:00"
        normalize_datetime_string("2021-04-13 18:10:00") -> "2021-04-13T18:10:00"
    """
    if not dt_string or not isinstance(dt_string, str):
        return ""
    
    dt_clean = re.sub(r'(\+\d{2}:\d{2}|Z)$', '', dt_string.strip())
    
    if ' ' in dt_clean and 'T' not in dt_clean:
        dt_clean = dt_clean.replace(' ', 'T')
    
    if len(dt_clean) >= 19 and dt_clean[10] != 'T':
        if len(dt_clean.split()) == 2:
            date_part, time_part = dt_clean.split()
            dt_clean = f"{date_part}T{time_part}"

    if '.' in dt_clean:
        dt_clean = dt_clean.split('.')[0]
    
    return dt_clean
