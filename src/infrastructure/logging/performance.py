"""
Performance logging utilities and decorators.
Provides automatic timing and resource tracking for operations.
"""

import time
import functools
import psutil
import threading
from typing import Callable, Any, Optional, Dict
from contextlib import contextmanager
from datetime import datetime

from .base import LoggerFactory


class PerformanceTimer:
    """
    Context manager and decorator for timing operations.
    """
    
    def __init__(self, 
                 operation_name: str,
                 logger_name: str = 'trading.performance',
                 include_memory: bool = False,
                 include_cpu: bool = False):
        """
        Initialize performance timer.
        
        Args:
            operation_name: Name of operation being timed
            logger_name: Logger to use
            include_memory: Whether to track memory usage
            include_cpu: Whether to track CPU usage
        """
        self.operation_name = operation_name
        self.logger = LoggerFactory.get_logger(logger_name)
        self.include_memory = include_memory
        self.include_cpu = include_cpu
        
        self.start_time = None
        self.end_time = None
        self.duration = None
        
        self.start_memory = None
        self.end_memory = None
        self.memory_delta = None
        
        self.start_cpu = None
        self.end_cpu = None
        self.cpu_percent = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        
        if self.include_memory:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if self.include_cpu:
            self.start_cpu = time.process_time()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log results."""
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        
        metrics = {
            'duration_seconds': self.duration,
            'duration_ms': self.duration * 1000
        }
        
        if self.include_memory:
            process = psutil.Process()
            self.end_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.memory_delta = self.end_memory - self.start_memory
            metrics['memory_start_mb'] = self.start_memory
            metrics['memory_end_mb'] = self.end_memory
            metrics['memory_delta_mb'] = self.memory_delta
        
        if self.include_cpu:
            self.end_cpu = time.process_time()
            self.cpu_percent = (self.end_cpu - self.start_cpu) / self.duration * 100
            metrics['cpu_percent'] = self.cpu_percent
        
        # Add exception info if operation failed
        if exc_type:
            metrics['failed'] = True
            metrics['exception'] = exc_type.__name__
        else:
            metrics['failed'] = False
        
        self.logger.log_performance(
            self.operation_name,
            self.duration,
            **metrics
        )


def log_performance(operation_name: Optional[str] = None,
                   include_memory: bool = False,
                   include_cpu: bool = False,
                   log_args: bool = False,
                   log_result: bool = False):
    """
    Decorator for automatic performance logging.
    
    Args:
        operation_name: Name of operation (uses function name if not provided)
        include_memory: Whether to track memory usage
        include_cpu: Whether to track CPU usage
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        
    Returns:
        Decorated function
        
    Example:
        @log_performance(include_memory=True)
        def process_data(ticker):
            # Function implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            timer = PerformanceTimer(
                op_name,
                include_memory=include_memory,
                include_cpu=include_cpu
            )
            
            extra_context = {}
            
            if log_args:
                extra_context['args'] = str(args)[:100]  # Truncate long args
                extra_context['kwargs'] = str(kwargs)[:100]
            
            with timer:
                result = func(*args, **kwargs)
            
            if log_result:
                extra_context['result'] = str(result)[:100]  # Truncate long results
            
            if extra_context:
                timer.logger.info(
                    f"Performance context for {op_name}",
                    extra=extra_context
                )
            
            return result
        
        return wrapper
    return decorator


class OperationTracker:
    """
    Tracks multiple operations and provides aggregated statistics.
    """
    
    def __init__(self, name: str = "OperationTracker"):
        """
        Initialize operation tracker.
        
        Args:
            name: Tracker name
        """
        self.name = name
        self.operations = {}
        self._lock = threading.Lock()
        self.logger = LoggerFactory.get_performance_logger()
    
    @contextmanager
    def track(self, operation_name: str):
        """
        Track an operation.
        
        Args:
            operation_name: Name of operation
            
        Yields:
            Operation context
        """
        start_time = time.perf_counter()
        
        try:
            yield
            success = True
        except Exception:
            success = False
            raise
        finally:
            duration = time.perf_counter() - start_time
            
            with self._lock:
                if operation_name not in self.operations:
                    self.operations[operation_name] = {
                        'count': 0,
                        'total_time': 0,
                        'min_time': float('inf'),
                        'max_time': 0,
                        'success_count': 0,
                        'failure_count': 0
                    }
                
                stats = self.operations[operation_name]
                stats['count'] += 1
                stats['total_time'] += duration
                stats['min_time'] = min(stats['min_time'], duration)
                stats['max_time'] = max(stats['max_time'], duration)
                
                if success:
                    stats['success_count'] += 1
                else:
                    stats['failure_count'] += 1
    
    def get_statistics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for operations.
        
        Args:
            operation_name: Specific operation or None for all
            
        Returns:
            Statistics dictionary
        """
        with self._lock:
            if operation_name:
                stats = self.operations.get(operation_name, {})
                if stats and stats['count'] > 0:
                    stats['avg_time'] = stats['total_time'] / stats['count']
                    stats['success_rate'] = stats['success_count'] / stats['count']
                return stats
            else:
                all_stats = {}
                for name, stats in self.operations.items():
                    if stats['count'] > 0:
                        stats_copy = stats.copy()
                        stats_copy['avg_time'] = stats['total_time'] / stats['count']
                        stats_copy['success_rate'] = stats['success_count'] / stats['count']
                        all_stats[name] = stats_copy
                return all_stats
    
    def log_statistics(self):
        """Log all collected statistics."""
        stats = self.get_statistics()
        
        for operation_name, op_stats in stats.items():
            self.logger.info(
                f"Operation statistics: {operation_name}",
                extra={
                    'statistics': {
                        'operation': operation_name,
                        'count': op_stats['count'],
                        'avg_time_seconds': op_stats['avg_time'],
                        'min_time_seconds': op_stats['min_time'],
                        'max_time_seconds': op_stats['max_time'],
                        'total_time_seconds': op_stats['total_time'],
                        'success_rate': op_stats['success_rate']
                    }
                }
            )
    
    def reset(self):
        """Reset all statistics."""
        with self._lock:
            self.operations.clear()


class BatchOperationTimer:
    """
    Times batch operations and provides throughput metrics.
    """
    
    def __init__(self, operation_name: str, batch_size: int):
        """
        Initialize batch operation timer.
        
        Args:
            operation_name: Name of batch operation
            batch_size: Number of items in batch
        """
        self.operation_name = operation_name
        self.batch_size = batch_size
        self.logger = LoggerFactory.get_performance_logger()
        
        self.start_time = None
        self.items_processed = 0
    
    def __enter__(self):
        """Start timing batch operation."""
        self.start_time = time.perf_counter()
        return self
    
    def update(self, items_processed: int):
        """
        Update number of items processed.
        
        Args:
            items_processed: Additional items processed
        """
        self.items_processed += items_processed
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Calculate and log batch metrics."""
        duration = time.perf_counter() - self.start_time
        
        if self.items_processed == 0:
            self.items_processed = self.batch_size
        
        throughput = self.items_processed / duration if duration > 0 else 0
        
        self.logger.log_performance(
            self.operation_name,
            duration,
            batch_size=self.batch_size,
            items_processed=self.items_processed,
            throughput_per_second=throughput,
            avg_time_per_item_ms=(duration / self.items_processed * 1000) if self.items_processed > 0 else 0,
            failed=exc_type is not None
        )


def measure_latency(func: Callable) -> Callable:
    """
    Decorator to measure function latency percentiles.
    
    Args:
        func: Function to measure
        
    Returns:
        Decorated function
    """
    latencies = []
    lock = threading.Lock()
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            latency = (time.perf_counter() - start) * 1000  # Convert to ms
            
            with lock:
                latencies.append(latency)
                
                # Keep only last 1000 measurements
                if len(latencies) > 1000:
                    latencies.pop(0)
                
                # Log percentiles every 100 calls
                if len(latencies) % 100 == 0:
                    sorted_latencies = sorted(latencies)
                    n = len(sorted_latencies)
                    
                    logger = LoggerFactory.get_performance_logger()
                    logger.info(
                        f"Latency percentiles for {func.__name__}",
                        extra={
                            'latency_percentiles': {
                                'p50': sorted_latencies[int(n * 0.5)],
                                'p75': sorted_latencies[int(n * 0.75)],
                                'p90': sorted_latencies[int(n * 0.9)],
                                'p95': sorted_latencies[int(n * 0.95)],
                                'p99': sorted_latencies[int(n * 0.99)] if n > 100 else None,
                                'min': sorted_latencies[0],
                                'max': sorted_latencies[-1],
                                'sample_size': n
                            }
                        }
                    )
    
    return wrapper


# Global operation tracker for shared use
global_tracker = OperationTracker("GlobalOperations")