import logging
import os
import sys
from datetime import datetime

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

# Cached default log directory
_DEFAULT_LOG_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../logs"
)

def _setup_structlog(log_dir: str, console_level: int, file_level: int):
    """Configure structlog processors and standard library routing."""
    
    # 1. Configure standard logging to handle structlog's output
    if not logging.root.handlers:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.jsonl")
        
        # File handler gets raw JSON strings directly from structlog
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(logging.Formatter('%(message)s'))

        # Console handler gets pretty colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        
        logging.root.setLevel(min(console_level, file_level))
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)

    # 2. Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    def _console_renderer(logger, method, event_dict):
        """Output plain message text to console — no timestamp or level prefix."""
        level = event_dict.get("level", "info").lower()
        msg = event_dict.get("event", "")
        if level in ("warning", "warn"):
            return f"[WARNING] {msg}"
        elif level in ("error", "critical"):
            return f"[ERROR] {msg}"
        return msg

    # Configure the formatters specifically for the handlers we just made
    formatter_json = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
    )
    formatter_console = structlog.stdlib.ProcessorFormatter(
        processor=_console_renderer,
    )
    
    # Map formatters to standard handlers
    handlers = logging.root.handlers
    if len(handlers) >= 2:
        handlers[0].setFormatter(formatter_json)
        handlers[1].setFormatter(formatter_console)

def get_logger(name: str, log_dir: str = None, 
               console_level: int = logging.INFO,
               file_level: int = logging.DEBUG):
    """
    Get a configured structlog instance. Fallbacks to standard logging if structlog is missing.
    """
    if not STRUCTLOG_AVAILABLE:
        # Fallback to standard logging for backward compatibility if structlog isn't installed
        logger = logging.getLogger(name)
        if hasattr(logger, "is_setup"): return logger
        
        logger.setLevel(logging.INFO)
        log_dir = log_dir or _DEFAULT_LOG_DIR
        os.makedirs(log_dir, exist_ok=True)
        
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch.setFormatter(fmt)
        
        log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(file_level)
        fh.setFormatter(fmt)
        
        logger.addHandler(ch)
        logger.addHandler(fh)
        logger.is_setup = True
        return logger

    # Initialize structlog
    log_dir = log_dir or _DEFAULT_LOG_DIR
    _setup_structlog(log_dir, console_level, file_level)
    return structlog.get_logger(name)
