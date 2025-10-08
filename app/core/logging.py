"""
Logging configuration for the application.
Sets up structured logging with appropriate handlers and formatters.
"""

import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from app.config.settings import settings


def timed_print(*args, **kwargs):
    """
    Print function that adds a timestamp to each message.
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    """
    # Use datetime for proper microsecond support
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}]", *args, **kwargs)


def setup_logging():
    """Set up application logging configuration."""
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler with proper encoding for Windows
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Set encoding for Windows to handle Unicode characters
    if sys.platform == "win32":
        # Use UTF-8 encoding for console output
        console_handler.setStream(sys.stdout)
        # Ensure stdout uses UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    
    root_logger.addHandler(console_handler)
    
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configuration initialized")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: The logger name
        
    Returns:
        logging.Logger: The configured logger instance
    """
    return logging.getLogger(name)
