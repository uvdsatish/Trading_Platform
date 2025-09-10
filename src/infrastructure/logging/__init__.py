"""
Logging infrastructure for the trading platform.

Provides structured logging, error handling, performance tracking,
and audit capabilities with automatic rotation and metrics collection.

Usage Examples:
    # Basic logging
    from src.infrastructure.logging import get_logger
    
    logger = get_logger(__name__)
    logger.info("Processing market data", extra={'ticker': 'AAPL'})
    
    # Performance logging
    from src.infrastructure.logging import log_performance
    
    @log_performance(include_memory=True)
    def process_data():
        # Your code here
        pass
    
    # Request context
    from src.infrastructure.logging import set_request_context
    
    with set_request_context(user_id="user123"):
        # All logs within this context will include user_id
        logger.info("User action performed")
"""

from .base import (
    TradingLogger,
    LogContext,
    LoggerFactory,
    generate_request_id,
    set_request_context,
    request_id_var,
    user_id_var,
    session_id_var
)

from .formatters import (
    JSONFormatter,
    ColoredConsoleFormatter,
    SimpleFormatter,
    CSVFormatter,
    AuditFormatter,
    get_formatter
)

from .handlers import (
    RotatingJSONFileHandler,
    TimedRotatingJSONFileHandler,
    BufferedAsyncHandler,
    MultiFileHandler,
    MetricsHandler
)

from .config import (
    setup_logging,
    get_logging_config,
    init
)

from .performance import (
    PerformanceTimer,
    log_performance,
    OperationTracker,
    BatchOperationTimer,
    measure_latency,
    global_tracker
)

# Convenience functions
def get_logger(name: str) -> TradingLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        TradingLogger instance
    """
    return LoggerFactory.get_logger(name)


def get_data_logger() -> TradingLogger:
    """Get logger for data collection operations."""
    return LoggerFactory.get_data_logger()


def get_technical_logger() -> TradingLogger:
    """Get logger for technical analysis operations."""
    return LoggerFactory.get_technical_logger()


def get_internals_logger() -> TradingLogger:
    """Get logger for market internals operations."""
    return LoggerFactory.get_internals_logger()


def get_trading_logger() -> TradingLogger:
    """Get logger for trading operations."""
    return LoggerFactory.get_trading_logger()


def get_backtesting_logger() -> TradingLogger:
    """Get logger for backtesting operations."""
    return LoggerFactory.get_backtesting_logger()


def get_performance_logger() -> TradingLogger:
    """Get logger for performance monitoring."""
    return LoggerFactory.get_performance_logger()


# Initialize logging on import
try:
    init()
except Exception:
    # Logging initialization failed, but don't break imports
    pass


__all__ = [
    # Main logger classes
    'TradingLogger',
    'LogContext',
    'LoggerFactory',
    
    # Convenience functions
    'get_logger',
    'get_data_logger',
    'get_technical_logger',
    'get_internals_logger',
    'get_trading_logger',
    'get_backtesting_logger',
    'get_performance_logger',
    
    # Context management
    'generate_request_id',
    'set_request_context',
    'request_id_var',
    'user_id_var',
    'session_id_var',
    
    # Formatters
    'JSONFormatter',
    'ColoredConsoleFormatter',
    'SimpleFormatter',
    'CSVFormatter',
    'AuditFormatter',
    'get_formatter',
    
    # Handlers
    'RotatingJSONFileHandler',
    'TimedRotatingJSONFileHandler',
    'BufferedAsyncHandler',
    'MultiFileHandler',
    'MetricsHandler',
    
    # Configuration
    'setup_logging',
    'get_logging_config',
    
    # Performance tracking
    'PerformanceTimer',
    'log_performance',
    'OperationTracker',
    'BatchOperationTimer',
    'measure_latency',
    'global_tracker',
]