"""
Base logging infrastructure for the trading platform.
Provides structured logging with context management.
"""

import logging
import sys
import json
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, Union
from contextvars import ContextVar
from pathlib import Path
import threading
import uuid


# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)


class TradingLogger:
    """
    Enhanced logger for the trading platform with structured logging support.
    """
    
    _instances: Dict[str, 'TradingLogger'] = {}
    _lock = threading.Lock()
    
    def __new__(cls, name: str, *args, **kwargs):
        """Implement singleton pattern per logger name."""
        with cls._lock:
            if name not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[name] = instance
            return cls._instances[name]
    
    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize trading logger.
        
        Args:
            name: Logger name (usually module name)
            level: Logging level
        """
        if hasattr(self, '_initialized'):
            return
            
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._initialized = True
        
        # Add default context
        self._default_context = {
            'logger_name': name,
            'platform': 'trading_platform',
            'version': '2.0.0'
        }
    
    def _get_context(self) -> Dict[str, Any]:
        """
        Get current logging context including correlation IDs.
        
        Returns:
            Context dictionary
        """
        context = self._default_context.copy()
        
        # Add correlation IDs if set
        request_id = request_id_var.get()
        if request_id:
            context['request_id'] = request_id
            
        user_id = user_id_var.get()
        if user_id:
            context['user_id'] = user_id
            
        session_id = session_id_var.get()
        if session_id:
            context['session_id'] = session_id
            
        # Add thread info
        context['thread_id'] = threading.current_thread().ident
        context['thread_name'] = threading.current_thread().name
        
        return context
    
    def _format_message(self, message: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format log message with context.
        
        Args:
            message: Log message
            extra: Additional context
            
        Returns:
            Formatted log entry
        """
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'message': message,
            'context': self._get_context()
        }
        
        if extra:
            log_entry['extra'] = extra
            
        return log_entry
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        extra = kwargs.pop('extra', {})
        log_data = self._format_message(message, extra)
        self.logger.debug(json.dumps(log_data), **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        extra = kwargs.pop('extra', {})
        log_data = self._format_message(message, extra)
        self.logger.info(json.dumps(log_data), **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        extra = kwargs.pop('extra', {})
        log_data = self._format_message(message, extra)
        self.logger.warning(json.dumps(log_data), **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """
        Log error message with exception details.
        
        Args:
            message: Error message
            exception: Optional exception object
            **kwargs: Additional arguments
        """
        extra = kwargs.pop('extra', {})
        
        if exception:
            extra['exception'] = {
                'type': type(exception).__name__,
                'message': str(exception),
                'traceback': traceback.format_exc()
            }
            
        log_data = self._format_message(message, extra)
        self.logger.error(json.dumps(log_data), **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """
        Log critical message with exception details.
        
        Args:
            message: Critical message
            exception: Optional exception object
            **kwargs: Additional arguments
        """
        extra = kwargs.pop('extra', {})
        
        if exception:
            extra['exception'] = {
                'type': type(exception).__name__,
                'message': str(exception),
                'traceback': traceback.format_exc()
            }
            
        log_data = self._format_message(message, extra)
        self.logger.critical(json.dumps(log_data), **kwargs)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """
        Log performance metrics.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            **kwargs: Additional metrics
        """
        extra = {
            'performance': {
                'operation': operation,
                'duration_seconds': duration,
                'duration_ms': duration * 1000,
                **kwargs
            }
        }
        
        message = f"Performance: {operation} took {duration:.3f}s"
        log_data = self._format_message(message, extra)
        self.logger.info(json.dumps(log_data))
    
    def log_trade(self, action: str, ticker: str, quantity: int, 
                  price: float, **kwargs):
        """
        Log trading activity.
        
        Args:
            action: Trade action (BUY, SELL, etc.)
            ticker: Stock ticker
            quantity: Number of shares
            price: Price per share
            **kwargs: Additional trade details
        """
        extra = {
            'trade': {
                'action': action,
                'ticker': ticker,
                'quantity': quantity,
                'price': price,
                'total_value': quantity * price,
                'timestamp': datetime.utcnow().isoformat(),
                **kwargs
            }
        }
        
        message = f"Trade: {action} {quantity} {ticker} @ ${price}"
        log_data = self._format_message(message, extra)
        self.logger.info(json.dumps(log_data))
    
    def log_data_fetch(self, source: str, ticker: str, 
                       records: int, duration: float, **kwargs):
        """
        Log data fetching operations.
        
        Args:
            source: Data source (IQFeed, Yahoo, etc.)
            ticker: Stock ticker
            records: Number of records fetched
            duration: Fetch duration in seconds
            **kwargs: Additional details
        """
        extra = {
            'data_fetch': {
                'source': source,
                'ticker': ticker,
                'records': records,
                'duration_seconds': duration,
                'records_per_second': records / duration if duration > 0 else 0,
                **kwargs
            }
        }
        
        message = f"Data fetch: {records} records for {ticker} from {source}"
        log_data = self._format_message(message, extra)
        self.logger.info(json.dumps(log_data))


class LogContext:
    """
    Context manager for setting logging context variables.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize log context.
        
        Args:
            **kwargs: Context variables to set
        """
        self.context = kwargs
        self.tokens = {}
    
    def __enter__(self):
        """Set context variables."""
        if 'request_id' in self.context:
            self.tokens['request_id'] = request_id_var.set(self.context['request_id'])
        if 'user_id' in self.context:
            self.tokens['user_id'] = user_id_var.set(self.context['user_id'])
        if 'session_id' in self.context:
            self.tokens['session_id'] = session_id_var.set(self.context['session_id'])
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Reset context variables."""
        for key, token in self.tokens.items():
            if key == 'request_id':
                request_id_var.reset(token)
            elif key == 'user_id':
                user_id_var.reset(token)
            elif key == 'session_id':
                session_id_var.reset(token)


class LoggerFactory:
    """
    Factory for creating domain-specific loggers.
    """
    
    @staticmethod
    def get_logger(name: str, level: Optional[int] = None) -> TradingLogger:
        """
        Get or create a logger.
        
        Args:
            name: Logger name
            level: Optional logging level
            
        Returns:
            TradingLogger instance
        """
        if level is None:
            # Determine level based on environment
            from src.config import get_config_service
            config = get_config_service()
            level_str = config.get_setting('logging.level', 'INFO')
            level = getattr(logging, level_str.upper())
            
        return TradingLogger(name, level)
    
    @staticmethod
    def get_data_logger() -> TradingLogger:
        """Get logger for data collection operations."""
        return LoggerFactory.get_logger('trading.data_collection')
    
    @staticmethod
    def get_technical_logger() -> TradingLogger:
        """Get logger for technical analysis operations."""
        return LoggerFactory.get_logger('trading.technical_analysis')
    
    @staticmethod
    def get_internals_logger() -> TradingLogger:
        """Get logger for market internals operations."""
        return LoggerFactory.get_logger('trading.market_internals')
    
    @staticmethod
    def get_trading_logger() -> TradingLogger:
        """Get logger for trading operations."""
        return LoggerFactory.get_logger('trading.execution')
    
    @staticmethod
    def get_backtesting_logger() -> TradingLogger:
        """Get logger for backtesting operations."""
        return LoggerFactory.get_logger('trading.backtesting')
    
    @staticmethod
    def get_performance_logger() -> TradingLogger:
        """Get logger for performance monitoring."""
        return LoggerFactory.get_logger('trading.performance')


def generate_request_id() -> str:
    """
    Generate a unique request ID.
    
    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def set_request_context(request_id: Optional[str] = None,
                        user_id: Optional[str] = None,
                        session_id: Optional[str] = None) -> LogContext:
    """
    Set logging context for request tracking.
    
    Args:
        request_id: Request ID (generated if not provided)
        user_id: User identifier
        session_id: Session identifier
        
    Returns:
        LogContext manager
    """
    if request_id is None:
        request_id = generate_request_id()
        
    return LogContext(
        request_id=request_id,
        user_id=user_id,
        session_id=session_id
    )