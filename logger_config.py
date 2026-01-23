"""
Centralized logging configuration for KnowVec RAG Pipeline.

This module provides a standardized logging setup with:
- Environment-based configuration (LOG_LEVEL, LOG_FORMAT)
- Production-optimized condensed formatting
- Console and file output with rotation
- Structured log formatting for log aggregation
- Performance tracking utilities

Environment Variables:
- LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (default: INFO)
- LOG_FORMAT: standard, json, production (default: production)
- ENABLE_FILE_LOGGING: true/false (default: true)
- LOG_DIR: Directory for log files (default: logs)
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs logs in structured JSON format for log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured data."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class ProductionFormatter(logging.Formatter):
    """
    Production-optimized formatter with condensed key=value output.

    Output format:
    2026-01-22 16:20:48 | INFO  | module | message | key1=val1 key2=val2

    This format is:
    - Easy to read in terminals
    - Easy to parse with log aggregation tools
    - Condensed to reduce log volume
    """

    LEVEL_COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record in production-optimized format."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname.ljust(5)

        # Shorten module name for cleaner output
        module = record.module
        if len(module) > 20:
            module = module[:17] + "..."
        module = module.ljust(20)

        message = record.getMessage()

        # Apply colors if enabled
        if self.use_colors:
            color = self.LEVEL_COLORS.get(record.levelname, '')
            level = f"{color}{level}{self.RESET}"

        # Build base log line
        log_line = f"{timestamp} | {level} | {module} | {message}"

        # Add exception info if present
        if record.exc_info:
            log_line += f"\n{self.formatException(record.exc_info)}"

        return log_line


class StandardFormatter(logging.Formatter):
    """Standard human-readable formatter with full details (for development)."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


def setup_logger(
    name: str = "knowvec",
    log_level: Optional[str] = None,
    log_format: str = "production",
    enable_file_logging: bool = True,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Set up and configure a logger with console and optional file output.

    Args:
        name: Logger name (typically __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                  If None, reads from LOG_LEVEL env var (default: INFO)
        log_format: Format type - 'production', 'standard', or 'json'
                   Reads from LOG_FORMAT env var (default: production)
                   - production: Condensed format for production use
                   - standard: Verbose format for development/debugging
                   - json: Structured JSON for log aggregation
        enable_file_logging: Whether to enable file logging
                            Reads from ENABLE_FILE_LOGGING env var (default: True)
        log_dir: Directory for log files (default: logs)
                Reads from LOG_DIR env var

    Returns:
        Configured logger instance
    """
    # Get configuration from environment variables with fallbacks
    log_level = log_level or os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", log_format).lower()
    enable_file_logging = os.getenv("ENABLE_FILE_LOGGING", str(enable_file_logging)).lower() == "true"
    log_dir = os.getenv("LOG_DIR", log_dir)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Choose formatter based on format type
    if log_format == "json":
        formatter = StructuredFormatter()
    elif log_format == "standard":
        formatter = StandardFormatter()
    else:  # Default to production
        formatter = ProductionFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    if enable_file_logging:
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # Main log file
            file_handler = logging.handlers.RotatingFileHandler(
                log_path / "knowvec.log",
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8"
            )
            file_handler.setLevel(getattr(logging, log_level, logging.INFO))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # Error log file
            error_handler = logging.handlers.RotatingFileHandler(
                log_path / "knowvec_errors.log",
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8"
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)

        except Exception as e:
            logger.warning(f"Failed to set up file logging: {e}")

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the standard configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    # Check if logger already exists and is configured
    logger = logging.getLogger(name)

    # If logger has no handlers, set it up
    if not logger.handlers:
        return setup_logger(name)

    return logger


class LoggerContext:
    """Context manager for temporary logger configuration."""

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.original_level = None

    def __enter__(self):
        self.original_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function entry and exit.

    Usage:
        @log_function_call(logger)
        def my_function(arg1, arg2):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func.__name__} successfully")
                return result
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator


# Initialize root logger
root_logger = setup_logger()


if __name__ == "__main__":
    # Test logging configuration
    test_logger = get_logger(__name__)

    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")

    try:
        raise ValueError("Test exception")
    except ValueError:
        test_logger.error("Caught test exception", exc_info=True)

    print("\nLogger configuration test complete!")
