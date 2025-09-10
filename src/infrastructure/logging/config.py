"""
Logging configuration for the trading platform.
Sets up handlers, formatters, and loggers based on environment.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from src.config import get_config_service
from .formatters import (
    JSONFormatter,
    ColoredConsoleFormatter,
    SimpleFormatter,
    AuditFormatter
)
from .handlers import (
    RotatingJSONFileHandler,
    TimedRotatingJSONFileHandler,
    BufferedAsyncHandler,
    MultiFileHandler,
    MetricsHandler
)


def setup_logging(
    environment: Optional[str] = None,
    log_dir: str = 'logs',
    enable_console: bool = True,
    enable_file: bool = True,
    enable_metrics: bool = True
) -> None:
    """
    Set up logging configuration for the trading platform.
    
    Args:
        environment: Environment (development, production, etc.)
        log_dir: Directory for log files
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
        enable_metrics: Whether to enable metrics collection
    """
    # Get configuration
    config = get_config_service()
    
    if environment is None:
        environment = config.environment
    
    # Determine log level
    log_level = config.get_setting('logging.level', 'INFO')
    log_level = getattr(logging, log_level.upper())
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    if enable_console:
        console_handler = _create_console_handler(environment)
        root_logger.addHandler(console_handler)
    
    # Add file handlers
    if enable_file:
        file_handlers = _create_file_handlers(log_path, environment)
        for handler in file_handlers:
            root_logger.addHandler(handler)
    
    # Add metrics handler
    if enable_metrics:
        metrics_handler = MetricsHandler(log_path / 'metrics.json')
        metrics_handler.setLevel(logging.INFO)
        root_logger.addHandler(metrics_handler)
    
    # Configure specific loggers
    _configure_domain_loggers(log_level)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging initialized for environment: {environment}",
        extra={'environment': environment, 'log_level': logging.getLevelName(log_level)}
    )


def _create_console_handler(environment: str) -> logging.Handler:
    """
    Create console handler based on environment.
    
    Args:
        environment: Current environment
        
    Returns:
        Console handler
    """
    console_handler = logging.StreamHandler(sys.stdout)
    
    if environment == 'development':
        # Colored output for development
        formatter = ColoredConsoleFormatter(include_context=True)
    else:
        # JSON output for production (easier to parse)
        formatter = JSONFormatter(include_traceback=False)
    
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG if environment == 'development' else logging.INFO)
    
    return console_handler


def _create_file_handlers(log_path: Path, environment: str) -> list:
    """
    Create file handlers for different log types.
    
    Args:
        log_path: Base path for log files
        environment: Current environment
        
    Returns:
        List of file handlers
    """
    handlers = []
    
    # Main application log (rotating by size)
    app_handler = RotatingJSONFileHandler(
        log_path / 'application.log',
        maxBytes=10485760,  # 10MB
        backupCount=10,
        compress=True
    )
    app_handler.setFormatter(JSONFormatter())
    app_handler.setLevel(logging.INFO)
    handlers.append(app_handler)
    
    # Error log (daily rotation)
    error_handler = TimedRotatingJSONFileHandler(
        log_path / 'errors.log',
        when='midnight',
        backupCount=30,
        compress=True
    )
    error_handler.setFormatter(JSONFormatter(include_traceback=True))
    error_handler.setLevel(logging.ERROR)
    handlers.append(error_handler)
    
    # Audit log (never deleted, for compliance)
    audit_handler = logging.FileHandler(
        log_path / 'audit.log',
        encoding='utf-8'
    )
    audit_handler.setFormatter(AuditFormatter())
    audit_handler.setLevel(logging.INFO)
    audit_handler.addFilter(AuditLogFilter())
    handlers.append(audit_handler)
    
    # Performance log (for analysis)
    perf_handler = RotatingJSONFileHandler(
        log_path / 'performance.log',
        maxBytes=5242880,  # 5MB
        backupCount=5,
        compress=True
    )
    perf_handler.setFormatter(JSONFormatter())
    perf_handler.setLevel(logging.INFO)
    perf_handler.addFilter(PerformanceLogFilter())
    handlers.append(perf_handler)
    
    # Trade log (critical for record keeping)
    trade_handler = logging.FileHandler(
        log_path / 'trades.log',
        encoding='utf-8'
    )
    trade_handler.setFormatter(JSONFormatter())
    trade_handler.setLevel(logging.INFO)
    trade_handler.addFilter(TradeLogFilter())
    handlers.append(trade_handler)
    
    # Development debug log
    if environment == 'development':
        debug_handler = RotatingJSONFileHandler(
            log_path / 'debug.log',
            maxBytes=20971520,  # 20MB
            backupCount=3
        )
        debug_handler.setFormatter(SimpleFormatter(include_module_info=True))
        debug_handler.setLevel(logging.DEBUG)
        handlers.append(debug_handler)
    
    return handlers


def _configure_domain_loggers(default_level: int) -> None:
    """
    Configure domain-specific loggers.
    
    Args:
        default_level: Default logging level
    """
    # Data collection logger
    data_logger = logging.getLogger('trading.data_collection')
    data_logger.setLevel(default_level)
    
    # Technical analysis logger
    tech_logger = logging.getLogger('trading.technical_analysis')
    tech_logger.setLevel(default_level)
    
    # Market internals logger
    internals_logger = logging.getLogger('trading.market_internals')
    internals_logger.setLevel(default_level)
    
    # Trading execution logger
    trading_logger = logging.getLogger('trading.execution')
    trading_logger.setLevel(logging.INFO)  # Always log trades
    
    # Backtesting logger
    backtest_logger = logging.getLogger('trading.backtesting')
    backtest_logger.setLevel(default_level)
    
    # Performance logger
    perf_logger = logging.getLogger('trading.performance')
    perf_logger.setLevel(logging.INFO)  # Always log performance
    
    # Database logger (reduce verbosity)
    db_logger = logging.getLogger('src.infrastructure.database')
    db_logger.setLevel(logging.WARNING)
    
    # External library loggers (reduce noise)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('psycopg2').setLevel(logging.WARNING)


class AuditLogFilter(logging.Filter):
    """Filter for audit logs."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Check if record should be included in audit log.
        
        Args:
            record: Log record
            
        Returns:
            True if record should be included
        """
        # Include trade logs
        if 'trade' in str(record.msg).lower():
            return True
        
        # Include configuration changes
        if 'config' in str(record.msg).lower() and 'change' in str(record.msg).lower():
            return True
        
        # Include authentication/authorization
        if any(word in str(record.msg).lower() for word in ['auth', 'login', 'permission']):
            return True
        
        return False


class PerformanceLogFilter(logging.Filter):
    """Filter for performance logs."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Check if record should be included in performance log.
        
        Args:
            record: Log record
            
        Returns:
            True if record should be included
        """
        try:
            import json
            msg = json.loads(str(record.msg))
            return 'performance' in msg.get('extra', {})
        except (json.JSONDecodeError, AttributeError):
            return 'performance' in str(record.msg).lower()


class TradeLogFilter(logging.Filter):
    """Filter for trade logs."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Check if record should be included in trade log.
        
        Args:
            record: Log record
            
        Returns:
            True if record should be included
        """
        try:
            import json
            msg = json.loads(str(record.msg))
            return 'trade' in msg.get('extra', {})
        except (json.JSONDecodeError, AttributeError):
            return 'trade' in str(record.msg).lower()


def get_logging_config() -> Dict[str, Any]:
    """
    Get logging configuration dictionary for use with logging.config.dictConfig.
    
    Returns:
        Logging configuration dictionary
    """
    config = get_config_service()
    environment = config.environment
    log_level = config.get_setting('logging.level', 'INFO')
    
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                '()': JSONFormatter,
                'include_traceback': True
            },
            'console': {
                '()': ColoredConsoleFormatter if environment == 'development' else JSONFormatter,
                'include_context': True
            },
            'simple': {
                '()': SimpleFormatter,
                'include_module_info': True
            },
            'audit': {
                '()': AuditFormatter
            }
        },
        'filters': {
            'audit': {
                '()': AuditLogFilter
            },
            'performance': {
                '()': PerformanceLogFilter
            },
            'trade': {
                '()': TradeLogFilter
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'DEBUG' if environment == 'development' else 'INFO',
                'formatter': 'console',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                '()': RotatingJSONFileHandler,
                'filename': 'logs/application.log',
                'maxBytes': 10485760,
                'backupCount': 10,
                'formatter': 'json',
                'level': 'INFO'
            },
            'error_file': {
                '()': TimedRotatingJSONFileHandler,
                'filename': 'logs/errors.log',
                'when': 'midnight',
                'backupCount': 30,
                'formatter': 'json',
                'level': 'ERROR'
            },
            'audit_file': {
                'class': 'logging.FileHandler',
                'filename': 'logs/audit.log',
                'formatter': 'audit',
                'filters': ['audit'],
                'level': 'INFO'
            },
            'performance_file': {
                '()': RotatingJSONFileHandler,
                'filename': 'logs/performance.log',
                'maxBytes': 5242880,
                'backupCount': 5,
                'formatter': 'json',
                'filters': ['performance'],
                'level': 'INFO'
            },
            'trade_file': {
                'class': 'logging.FileHandler',
                'filename': 'logs/trades.log',
                'formatter': 'json',
                'filters': ['trade'],
                'level': 'INFO'
            }
        },
        'loggers': {
            'trading.data_collection': {
                'level': log_level,
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'trading.technical_analysis': {
                'level': log_level,
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'trading.market_internals': {
                'level': log_level,
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'trading.execution': {
                'level': 'INFO',
                'handlers': ['console', 'file', 'trade_file', 'audit_file'],
                'propagate': False
            },
            'trading.backtesting': {
                'level': log_level,
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'trading.performance': {
                'level': 'INFO',
                'handlers': ['console', 'performance_file'],
                'propagate': False
            }
        },
        'root': {
            'level': log_level,
            'handlers': ['console', 'file', 'error_file']
        }
    }


# Initialize logging on module import
def init():
    """Initialize logging with default settings."""
    try:
        setup_logging()
    except Exception as e:
        # Fall back to basic config if setup fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.error(f"Failed to initialize custom logging: {e}")