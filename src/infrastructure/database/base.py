"""
Base classes for database connection management.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Optional, Generator, Callable
import logging
import threading
import time
from datetime import datetime, timedelta


class IConnectionPool(ABC):
    """Abstract interface for database connection pools."""
    
    @abstractmethod
    def get_connection(self) -> Any:
        """Get a connection from the pool."""
        pass
    
    @abstractmethod
    def return_connection(self, connection: Any) -> None:
        """Return a connection to the pool."""
        pass
    
    @abstractmethod
    def close_all(self) -> None:
        """Close all connections in the pool."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check pool health."""
        pass


class BaseConnectionManager:
    """Base class for managing database connections with context manager support."""
    
    def __init__(self, pool: IConnectionPool):
        """
        Initialize connection manager.
        
        Args:
            pool: Connection pool instance
        """
        self.pool = pool
        self.logger = logging.getLogger(self.__class__.__name__)
        self._local = threading.local()
    
    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """
        Context manager for acquiring and releasing connections.
        
        Yields:
            Database connection
            
        Example:
            with manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
        """
        connection = None
        try:
            connection = self.pool.get_connection()
            yield connection
        except Exception as e:
            self.logger.error(f"Error during connection usage: {e}")
            raise
        finally:
            if connection:
                self.pool.return_connection(connection)
    
    @contextmanager
    def cursor(self, **kwargs) -> Generator[Any, None, None]:
        """
        Context manager for cursor with automatic cleanup.
        
        Args:
            **kwargs: Arguments to pass to cursor creation
            
        Yields:
            Database cursor
        """
        with self.get_connection() as conn:
            cursor = None
            try:
                cursor = conn.cursor(**kwargs)
                yield cursor
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Error during cursor operation: {e}")
                raise
            else:
                conn.commit()
            finally:
                if cursor:
                    cursor.close()


class ConnectionPoolMetrics:
    """Tracks metrics for connection pool monitoring."""
    
    def __init__(self):
        """Initialize metrics tracking."""
        self._lock = threading.Lock()
        self._metrics = {
            'connections_created': 0,
            'connections_destroyed': 0,
            'connections_acquired': 0,
            'connections_released': 0,
            'connections_failed': 0,
            'wait_time_total': 0.0,
            'wait_count': 0,
            'active_connections': 0,
            'idle_connections': 0,
            'pool_exhausted_count': 0,
            'health_check_failures': 0,
            'last_health_check': None,
            'uptime_start': datetime.now()
        }
    
    def increment(self, metric: str, value: int = 1) -> None:
        """
        Increment a metric counter.
        
        Args:
            metric: Metric name
            value: Value to increment by
        """
        with self._lock:
            if metric in self._metrics:
                self._metrics[metric] += value
    
    def set(self, metric: str, value: Any) -> None:
        """
        Set a metric value.
        
        Args:
            metric: Metric name
            value: Value to set
        """
        with self._lock:
            self._metrics[metric] = value
    
    def add_wait_time(self, wait_time: float) -> None:
        """
        Add connection wait time to metrics.
        
        Args:
            wait_time: Time waited for connection in seconds
        """
        with self._lock:
            self._metrics['wait_time_total'] += wait_time
            self._metrics['wait_count'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current metrics.
        
        Returns:
            Dictionary of metrics
        """
        with self._lock:
            stats = self._metrics.copy()
            
            # Calculate derived metrics
            if stats['wait_count'] > 0:
                stats['avg_wait_time'] = stats['wait_time_total'] / stats['wait_count']
            else:
                stats['avg_wait_time'] = 0.0
            
            # Calculate uptime
            stats['uptime_seconds'] = (datetime.now() - stats['uptime_start']).total_seconds()
            
            return stats


class ConnectionValidator:
    """Validates database connections before use."""
    
    def __init__(self, validation_query: str = "SELECT 1", 
                 validation_timeout: float = 5.0):
        """
        Initialize connection validator.
        
        Args:
            validation_query: Query to validate connection
            validation_timeout: Timeout for validation query
        """
        self.validation_query = validation_query
        self.validation_timeout = validation_timeout
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(self, connection: Any) -> bool:
        """
        Validate a database connection.
        
        Args:
            connection: Database connection to validate
            
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # Set a timeout for the validation query
            cursor = connection.cursor()
            cursor.execute(f"SET statement_timeout = {int(self.validation_timeout * 1000)}")
            cursor.execute(self.validation_query)
            cursor.fetchone()
            cursor.close()
            return True
        except Exception as e:
            self.logger.debug(f"Connection validation failed: {e}")
            return False
    
    def repair(self, connection: Any) -> bool:
        """
        Attempt to repair a connection.
        
        Args:
            connection: Database connection to repair
            
        Returns:
            True if repair successful, False otherwise
        """
        try:
            # Try to rollback any pending transaction
            connection.rollback()
            
            # Reset connection state
            connection.autocommit = False
            
            # Validate the repaired connection
            return self.validate(connection)
        except Exception as e:
            self.logger.debug(f"Connection repair failed: {e}")
            return False


class PooledConnection:
    """Wrapper for pooled database connections with metadata."""
    
    def __init__(self, connection: Any, pool_id: str):
        """
        Initialize pooled connection wrapper.
        
        Args:
            connection: Underlying database connection
            pool_id: Identifier for the pool this connection belongs to
        """
        self.connection = connection
        self.pool_id = pool_id
        self.created_at = datetime.now()
        self.last_used_at = datetime.now()
        self.use_count = 0
        self.in_use = False
        self.thread_id = None
    
    def acquire(self, thread_id: Optional[int] = None) -> None:
        """
        Mark connection as acquired.
        
        Args:
            thread_id: ID of thread acquiring the connection
        """
        self.in_use = True
        self.thread_id = thread_id or threading.current_thread().ident
        self.last_used_at = datetime.now()
        self.use_count += 1
    
    def release(self) -> None:
        """Mark connection as released."""
        self.in_use = False
        self.thread_id = None
    
    def age_seconds(self) -> float:
        """
        Get connection age in seconds.
        
        Returns:
            Age of connection in seconds
        """
        return (datetime.now() - self.created_at).total_seconds()
    
    def idle_seconds(self) -> float:
        """
        Get idle time in seconds.
        
        Returns:
            Time since last use in seconds
        """
        return (datetime.now() - self.last_used_at).total_seconds()
    
    def should_recycle(self, max_age: int = 3600, max_uses: int = 1000) -> bool:
        """
        Check if connection should be recycled.
        
        Args:
            max_age: Maximum age in seconds
            max_uses: Maximum number of uses
            
        Returns:
            True if connection should be recycled
        """
        return (self.age_seconds() > max_age or 
                self.use_count > max_uses)