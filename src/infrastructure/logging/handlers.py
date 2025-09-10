"""
Custom log handlers with rotation, aggregation, and monitoring capabilities.
"""

import logging
import logging.handlers
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import threading
import queue
import gzip
import shutil


class RotatingJSONFileHandler(logging.handlers.RotatingFileHandler):
    """
    Rotating file handler that writes JSON formatted logs.
    Automatically compresses rotated files.
    """
    
    def __init__(self, 
                 filename: str,
                 mode: str = 'a',
                 maxBytes: int = 10485760,  # 10MB default
                 backupCount: int = 10,
                 encoding: Optional[str] = 'utf-8',
                 compress: bool = True):
        """
        Initialize rotating JSON file handler.
        
        Args:
            filename: Log file path
            mode: File mode
            maxBytes: Maximum file size before rotation
            backupCount: Number of backup files to keep
            encoding: File encoding
            compress: Whether to compress rotated files
        """
        super().__init__(filename, mode, maxBytes, backupCount, encoding)
        self.compress = compress
    
    def doRollover(self):
        """Override to add compression of rotated files."""
        super().doRollover()
        
        if self.compress and self.backupCount > 0:
            # Compress the just-rotated file
            source = f"{self.baseFilename}.1"
            if os.path.exists(source):
                self._compress_file(source)
    
    def _compress_file(self, filepath: str):
        """
        Compress a log file using gzip.
        
        Args:
            filepath: Path to file to compress
        """
        try:
            with open(filepath, 'rb') as f_in:
                with gzip.open(f"{filepath}.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(filepath)
        except Exception as e:
            # Log compression failure but don't stop logging
            print(f"Failed to compress log file {filepath}: {e}")


class TimedRotatingJSONFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    Time-based rotating file handler for JSON logs.
    Rotates logs daily and compresses old files.
    """
    
    def __init__(self,
                 filename: str,
                 when: str = 'midnight',
                 interval: int = 1,
                 backupCount: int = 30,
                 encoding: Optional[str] = 'utf-8',
                 compress: bool = True):
        """
        Initialize timed rotating file handler.
        
        Args:
            filename: Log file path
            when: When to rotate ('midnight', 'H', 'D', 'W0'-'W6')
            interval: Rotation interval
            backupCount: Number of backup files to keep
            encoding: File encoding
            compress: Whether to compress rotated files
        """
        super().__init__(filename, when, interval, backupCount, encoding)
        self.compress = compress
    
    def doRollover(self):
        """Override to add compression of rotated files."""
        super().doRollover()
        
        if self.compress:
            # Find and compress the just-rotated file
            self._compress_old_logs()
    
    def _compress_old_logs(self):
        """Compress rotated log files."""
        dir_name, base_name = os.path.split(self.baseFilename)
        file_names = os.listdir(dir_name if dir_name else '.')
        
        for file_name in file_names:
            if file_name.startswith(base_name) and not file_name.endswith('.gz'):
                if file_name != base_name:  # Don't compress current file
                    file_path = os.path.join(dir_name, file_name)
                    try:
                        with open(file_path, 'rb') as f_in:
                            with gzip.open(f"{file_path}.gz", 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        os.remove(file_path)
                    except Exception:
                        pass


class BufferedAsyncHandler(logging.Handler):
    """
    Asynchronous buffered handler for high-performance logging.
    Buffers logs in memory and writes them in batches.
    """
    
    def __init__(self, 
                 target_handler: logging.Handler,
                 buffer_size: int = 1000,
                 flush_interval: float = 5.0):
        """
        Initialize buffered async handler.
        
        Args:
            target_handler: Handler to forward logs to
            buffer_size: Maximum buffer size before flush
            flush_interval: Maximum time between flushes (seconds)
        """
        super().__init__()
        self.target_handler = target_handler
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.last_flush = datetime.now()
        
        # Start background thread for flushing
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
    
    def emit(self, record: logging.LogRecord):
        """
        Add record to buffer.
        
        Args:
            record: Log record
        """
        try:
            self.buffer.put_nowait(record)
        except queue.Full:
            # Force flush if buffer is full
            self._flush()
            self.buffer.put_nowait(record)
    
    def _flush_loop(self):
        """Background thread that periodically flushes the buffer."""
        while not self._stop_event.is_set():
            self._stop_event.wait(self.flush_interval)
            self._flush()
    
    def _flush(self):
        """Flush buffered records to target handler."""
        records = []
        
        # Drain the buffer
        while not self.buffer.empty():
            try:
                records.append(self.buffer.get_nowait())
            except queue.Empty:
                break
        
        # Write all records
        for record in records:
            self.target_handler.emit(record)
        
        self.last_flush = datetime.now()
    
    def flush(self):
        """Force flush of buffer."""
        self._flush()
        self.target_handler.flush()
    
    def close(self):
        """Close handler and flush remaining records."""
        self._stop_event.set()
        self._flush()
        self.target_handler.close()
        super().close()


class MultiFileHandler(logging.Handler):
    """
    Handler that routes logs to different files based on criteria.
    Useful for separating error logs, audit logs, performance logs, etc.
    """
    
    def __init__(self, base_dir: str = 'logs'):
        """
        Initialize multi-file handler.
        
        Args:
            base_dir: Base directory for log files
        """
        super().__init__()
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create handlers for different log types
        self.handlers = {
            'error': self._create_handler('errors.log', logging.ERROR),
            'warning': self._create_handler('warnings.log', logging.WARNING),
            'info': self._create_handler('info.log', logging.INFO),
            'debug': self._create_handler('debug.log', logging.DEBUG),
            'trade': self._create_handler('trades.log', logging.INFO),
            'performance': self._create_handler('performance.log', logging.INFO),
            'audit': self._create_handler('audit.log', logging.INFO),
        }
    
    def _create_handler(self, filename: str, level: int) -> logging.Handler:
        """
        Create a rotating file handler.
        
        Args:
            filename: Log file name
            level: Logging level
            
        Returns:
            File handler
        """
        handler = RotatingJSONFileHandler(
            self.base_dir / filename,
            maxBytes=10485760,  # 10MB
            backupCount=10,
            compress=True
        )
        handler.setLevel(level)
        return handler
    
    def emit(self, record: logging.LogRecord):
        """
        Route record to appropriate handler.
        
        Args:
            record: Log record
        """
        # Try to parse message for log type
        log_type = 'info'  # default
        
        try:
            if record.levelno >= logging.ERROR:
                log_type = 'error'
            elif record.levelno >= logging.WARNING:
                log_type = 'warning'
            elif record.levelno == logging.DEBUG:
                log_type = 'debug'
            
            # Check for special log types in message
            if hasattr(record, 'msg'):
                msg_str = str(record.msg)
                if 'trade' in msg_str.lower():
                    log_type = 'trade'
                elif 'performance' in msg_str.lower():
                    log_type = 'performance'
                elif 'audit' in msg_str.lower():
                    log_type = 'audit'
        except Exception:
            pass
        
        # Route to appropriate handler
        handler = self.handlers.get(log_type, self.handlers['info'])
        handler.emit(record)
    
    def flush(self):
        """Flush all handlers."""
        for handler in self.handlers.values():
            handler.flush()
    
    def close(self):
        """Close all handlers."""
        for handler in self.handlers.values():
            handler.close()
        super().close()


class MetricsHandler(logging.Handler):
    """
    Handler that extracts metrics from logs for monitoring.
    Aggregates performance metrics, error counts, etc.
    """
    
    def __init__(self, metrics_file: str = 'logs/metrics.json'):
        """
        Initialize metrics handler.
        
        Args:
            metrics_file: Path to metrics file
        """
        super().__init__()
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'error_count': 0,
            'warning_count': 0,
            'trade_count': 0,
            'performance_samples': [],
            'error_types': {},
            'last_update': None
        }
        
        self._lock = threading.Lock()
        self._load_metrics()
    
    def _load_metrics(self):
        """Load existing metrics from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
            except Exception:
                pass
    
    def _save_metrics(self):
        """Save metrics to file."""
        try:
            self.metrics['last_update'] = datetime.utcnow().isoformat()
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
        except Exception:
            pass
    
    def emit(self, record: logging.LogRecord):
        """
        Extract metrics from log record.
        
        Args:
            record: Log record
        """
        with self._lock:
            # Count errors and warnings
            if record.levelno >= logging.ERROR:
                self.metrics['error_count'] += 1
                
                # Track error types
                error_type = record.msg[:50] if hasattr(record, 'msg') else 'unknown'
                self.metrics['error_types'][error_type] = \
                    self.metrics['error_types'].get(error_type, 0) + 1
                    
            elif record.levelno >= logging.WARNING:
                self.metrics['warning_count'] += 1
            
            # Extract performance metrics
            try:
                if hasattr(record, 'msg'):
                    msg = json.loads(str(record.msg))
                    
                    if 'performance' in msg.get('extra', {}):
                        perf = msg['extra']['performance']
                        self.metrics['performance_samples'].append({
                            'operation': perf.get('operation'),
                            'duration_ms': perf.get('duration_ms'),
                            'timestamp': datetime.utcnow().isoformat()
                        })
                        
                        # Keep only last 1000 samples
                        self.metrics['performance_samples'] = \
                            self.metrics['performance_samples'][-1000:]
                    
                    if 'trade' in msg.get('extra', {}):
                        self.metrics['trade_count'] += 1
                        
            except (json.JSONDecodeError, KeyError):
                pass
            
            # Periodically save metrics
            if self.metrics['error_count'] % 10 == 0:
                self._save_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.
        
        Returns:
            Metrics dictionary
        """
        with self._lock:
            return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset metrics counters."""
        with self._lock:
            self.metrics = {
                'error_count': 0,
                'warning_count': 0,
                'trade_count': 0,
                'performance_samples': [],
                'error_types': {},
                'last_update': None
            }
            self._save_metrics()
    
    def close(self):
        """Save metrics and close handler."""
        self._save_metrics()
        super().close()