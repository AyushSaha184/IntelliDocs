import logging
import os
from datetime import datetime

# Cached formatters (created once, reused many times)
_CONSOLE_FORMATTER = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

_FILE_FORMATTER = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Cached default log directory
_DEFAULT_LOG_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../logs"
)


def get_logger(name: str, log_dir: str = None, 
               console_level: int = logging.INFO,
               file_level: int = logging.DEBUG) -> logging.Logger:
    """
    Get a configured logger instance
    
    Args:
        name: Logger name (usually __name__)
        log_dir: Directory to save logs. Defaults to logs/
        console_level: Logging level for console output. Defaults to INFO
        file_level: Logging level for file output. Defaults to DEBUG
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handlers if they don't exist (avoid duplicate logs)
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = log_dir or _DEFAULT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(_CONSOLE_FORMATTER)
    
    # File handler
    log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(_FILE_FORMATTER)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
