"""
Log formatters for different output formats and environments.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
import traceback


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    Ideal for production environments and log aggregation systems.
    """
    
    def __init__(self, include_traceback: bool = True):
        """
        Initialize JSON formatter.
        
        Args:
            include_traceback: Whether to include full traceback in errors
        """
        super().__init__()
        self.include_traceback = include_traceback
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record
            
        Returns:
            JSON string
        """
        # Try to parse message as JSON first (for structured logs)
        try:
            log_data = json.loads(record.getMessage())
        except (json.JSONDecodeError, ValueError):
            # Fall back to creating structured log
            log_data = {
                'timestamp': datetime.utcfromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
        
        # Add exception info if present
        if record.exc_info and self.include_traceback:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data['extra'] = record.extra_fields
        
        return json.dumps(log_data, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """
    Colored console formatter for development environments.
    Makes logs human-readable with color coding.
    """
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def __init__(self, include_context: bool = False):
        """
        Initialize colored formatter.
        
        Args:
            include_context: Whether to include context information
        """
        super().__init__()
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.
        
        Args:
            record: Log record
            
        Returns:
            Formatted string with color codes
        """
        # Try to parse structured message
        try:
            log_data = json.loads(record.getMessage())
            message = log_data.get('message', record.getMessage())
            context = log_data.get('context', {})
            extra = log_data.get('extra', {})
        except (json.JSONDecodeError, ValueError):
            message = record.getMessage()
            context = {}
            extra = {}
        
        # Format timestamp
        timestamp = datetime.utcfromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Get color for level
        color = self.COLORS.get(record.levelname, '')
        
        # Build formatted message
        parts = [
            f"{color}{self.BOLD}[{record.levelname}]{self.RESET}",
            f"{timestamp}",
            f"{self.BOLD}{record.name}{self.RESET}",
            message
        ]
        
        # Add context if requested
        if self.include_context and context:
            request_id = context.get('request_id')
            if request_id:
                parts.append(f"[{request_id[:8]}]")
        
        # Add extra fields for specific log types
        if 'trade' in extra:
            trade = extra['trade']
            parts.append(
                f"🔄 {trade['action']} {trade['quantity']} "
                f"{trade['ticker']} @ ${trade['price']}"
            )
        elif 'performance' in extra:
            perf = extra['performance']
            parts.append(f"⚡ {perf['operation']}: {perf['duration_ms']:.2f}ms")
        elif 'data_fetch' in extra:
            fetch = extra['data_fetch']
            parts.append(
                f"📊 {fetch['source']}: {fetch['records']} records "
                f"in {fetch['duration_seconds']:.2f}s"
            )
        
        # Add exception if present
        if record.exc_info:
            parts.append(f"\n{color}{''.join(traceback.format_exception(*record.exc_info))}{self.RESET}")
        
        return ' '.join(parts)


class SimpleFormatter(logging.Formatter):
    """
    Simple formatter for file logging.
    Human-readable without colors.
    """
    
    def __init__(self, include_module_info: bool = True):
        """
        Initialize simple formatter.
        
        Args:
            include_module_info: Whether to include module information
        """
        if include_module_info:
            format_str = '%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s'
        else:
            format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            
        super().__init__(format_str, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record.
        
        Args:
            record: Log record
            
        Returns:
            Formatted string
        """
        # Try to extract message from structured log
        try:
            log_data = json.loads(record.getMessage())
            record.msg = log_data.get('message', record.getMessage())
            
            # Add important context
            context = log_data.get('context', {})
            request_id = context.get('request_id')
            if request_id:
                record.msg = f"[{request_id[:8]}] {record.msg}"
                
        except (json.JSONDecodeError, ValueError):
            pass
        
        return super().format(record)


class CSVFormatter(logging.Formatter):
    """
    CSV formatter for data analysis.
    Useful for importing logs into spreadsheets or databases.
    """
    
    def __init__(self, delimiter: str = ',', quote_char: str = '"'):
        """
        Initialize CSV formatter.
        
        Args:
            delimiter: Field delimiter
            quote_char: Quote character for fields with delimiters
        """
        super().__init__()
        self.delimiter = delimiter
        self.quote_char = quote_char
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as CSV.
        
        Args:
            record: Log record
            
        Returns:
            CSV formatted string
        """
        # Extract message and context
        try:
            log_data = json.loads(record.getMessage())
            message = log_data.get('message', record.getMessage())
            context = log_data.get('context', {})
        except (json.JSONDecodeError, ValueError):
            message = record.getMessage()
            context = {}
        
        # Prepare fields
        fields = [
            datetime.utcfromtimestamp(record.created).isoformat(),
            record.levelname,
            record.name,
            record.module,
            str(record.lineno),
            self._escape_csv(message),
            context.get('request_id', ''),
            context.get('user_id', ''),
            context.get('thread_name', '')
        ]
        
        return self.delimiter.join(fields)
    
    def _escape_csv(self, value: str) -> str:
        """
        Escape CSV field value.
        
        Args:
            value: Field value
            
        Returns:
            Escaped value
        """
        if self.delimiter in value or self.quote_char in value or '\n' in value:
            # Escape quotes and wrap in quotes
            value = value.replace(self.quote_char, self.quote_char + self.quote_char)
            return f'{self.quote_char}{value}{self.quote_char}'
        return value


class AuditFormatter(logging.Formatter):
    """
    Formatter for audit logs with compliance requirements.
    Includes all context and cannot be modified.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format audit log record.
        
        Args:
            record: Log record
            
        Returns:
            Immutable audit log entry
        """
        # Parse structured message
        try:
            log_data = json.loads(record.getMessage())
        except (json.JSONDecodeError, ValueError):
            log_data = {'message': record.getMessage()}
        
        # Build comprehensive audit entry
        audit_entry = {
            'audit_timestamp': datetime.utcnow().isoformat() + 'Z',
            'record_timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread,
            'thread_name': record.threadName,
            **log_data
        }
        
        # Add hash for integrity verification (simplified)
        import hashlib
        content = json.dumps(audit_entry, sort_keys=True, default=str)
        audit_entry['integrity_hash'] = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        return json.dumps(audit_entry, sort_keys=True, default=str)


def get_formatter(formatter_type: str = 'console', **kwargs) -> logging.Formatter:
    """
    Get formatter by type.
    
    Args:
        formatter_type: Type of formatter (console, json, simple, csv, audit)
        **kwargs: Formatter-specific arguments
        
    Returns:
        Formatter instance
    """
    formatters = {
        'console': ColoredConsoleFormatter,
        'json': JSONFormatter,
        'simple': SimpleFormatter,
        'csv': CSVFormatter,
        'audit': AuditFormatter
    }
    
    formatter_class = formatters.get(formatter_type, SimpleFormatter)
    return formatter_class(**kwargs)