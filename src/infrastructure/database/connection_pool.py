"""
PostgreSQL connection pool implementation with thread safety and resource management.
"""

import psycopg2
import psycopg2.pool
from psycopg2 import OperationalError, DatabaseError as PGDatabaseError
from typing import Any, Dict, Optional, List
import threading
import time
import logging
from contextlib import contextmanager
from datetime import datetime

from .base import (
    IConnectionPool, 
    BaseConnectionManager,
    ConnectionPoolMetrics,
    ConnectionValidator,
    PooledConnection
)
from .exceptions import (
    ConnectionPoolError,
    ConnectionPoolExhaustedError,
    ConnectionAcquisitionError,
    ConnectionValidationError,
    DatabaseConnectionLostError
)
from src.config import DatabaseConfig


class PostgreSQLConnectionPool(IConnectionPool):
    """
    Thread-safe PostgreSQL connection pool with advanced features.
    """
    
    def __init__(self, 
                 db_config: DatabaseConfig,
                 min_connections: Optional[int] = None,
                 max_connections: Optional[int] = None,
                 connection_timeout: float = 30.0,
                 idle_timeout: int = 600,
                 max_age: int = 3600,
                 max_uses: int = 1000,
                 pre_ping: bool = True):
        """
        Initialize PostgreSQL connection pool.
        
        Args:
            db_config: Database configuration
            min_connections: Minimum number of connections (default from config)
            max_connections: Maximum number of connections (default from config)
            connection_timeout: Timeout for acquiring connection in seconds
            idle_timeout: Time before idle connections are closed (seconds)
            max_age: Maximum age of connection before recycling (seconds)
            max_uses: Maximum uses before recycling connection
            pre_ping: Validate connection before checkout
        """
        self.db_config = db_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Get pool configuration
        pool_config = db_config.get_pool_config()
        self.min_connections = min_connections or pool_config.get('pool_size', 5)
        self.max_connections = max_connections or pool_config.get('max_overflow', 20)
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        self.max_age = max_age
        self.max_uses = max_uses
        self.pre_ping = pre_ping
        
        # Connection string
        self.connection_string = db_config.get_connection_string()
        
        # Thread safety
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        
        # Pool state
        self._connections: List[PooledConnection] = []
        self._available_connections: List[PooledConnection] = []
        self._in_use_connections: Dict[int, PooledConnection] = {}
        self._closed = False
        
        # Metrics and validation
        self.metrics = ConnectionPoolMetrics()
        self.validator = ConnectionValidator()
        
        # Initialize pool
        self._initialize_pool()
        
        # Start background maintenance thread
        self._start_maintenance_thread()
        
        self.logger.info(
            f"PostgreSQL connection pool initialized: "
            f"min={self.min_connections}, max={self.max_connections}"
        )
    
    def _initialize_pool(self) -> None:
        """Initialize the minimum number of connections."""
        try:
            for _ in range(self.min_connections):
                conn = self._create_connection()
                if conn:
                    self._available_connections.append(conn)
                    self._connections.append(conn)
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise ConnectionPoolError(f"Pool initialization failed: {e}")
    
    def _create_connection(self) -> PooledConnection:
        """
        Create a new database connection.
        
        Returns:
            Wrapped database connection
        """
        try:
            # Create raw psycopg2 connection
            raw_conn = psycopg2.connect(self.connection_string)
            raw_conn.autocommit = False
            
            # Wrap in PooledConnection
            pooled_conn = PooledConnection(raw_conn, id(self))
            
            self.metrics.increment('connections_created')
            self.logger.debug("Created new database connection")
            
            return pooled_conn
        except OperationalError as e:
            self.metrics.increment('connections_failed')
            self.logger.error(f"Failed to create connection: {e}")
            raise ConnectionAcquisitionError(f"Cannot create connection: {e}")
    
    def get_connection(self) -> Any:
        """
        Get a connection from the pool.
        
        Returns:
            Database connection
            
        Raises:
            ConnectionPoolExhaustedError: If no connections available
            ConnectionAcquisitionError: If connection cannot be acquired
        """
        if self._closed:
            raise ConnectionPoolError("Connection pool is closed")
        
        start_time = time.time()
        timeout_time = start_time + self.connection_timeout
        
        with self._lock:
            while True:
                # Try to get an available connection
                connection = self._get_available_connection()
                
                if connection:
                    # Validate connection if pre_ping is enabled
                    if self.pre_ping and not self._validate_connection(connection):
                        self._remove_connection(connection)
                        continue
                    
                    # Mark as in use
                    thread_id = threading.current_thread().ident
                    connection.acquire(thread_id)
                    self._in_use_connections[thread_id] = connection
                    
                    # Update metrics
                    wait_time = time.time() - start_time
                    self.metrics.add_wait_time(wait_time)
                    self.metrics.increment('connections_acquired')
                    self.metrics.set('active_connections', len(self._in_use_connections))
                    self.metrics.set('idle_connections', len(self._available_connections))
                    
                    return connection.connection
                
                # Try to create new connection if under max
                if len(self._connections) < self.max_connections:
                    try:
                        new_conn = self._create_connection()
                        if new_conn:
                            self._connections.append(new_conn)
                            thread_id = threading.current_thread().ident
                            new_conn.acquire(thread_id)
                            self._in_use_connections[thread_id] = new_conn
                            
                            self.metrics.increment('connections_acquired')
                            self.metrics.set('active_connections', len(self._in_use_connections))
                            
                            return new_conn.connection
                    except Exception as e:
                        self.logger.warning(f"Failed to create new connection: {e}")
                
                # Check timeout
                if time.time() >= timeout_time:
                    self.metrics.increment('pool_exhausted_count')
                    raise ConnectionPoolExhaustedError(
                        f"No connections available after {self.connection_timeout}s"
                    )
                
                # Wait for connection to become available
                remaining_time = timeout_time - time.time()
                if remaining_time > 0:
                    self._condition.wait(timeout=min(remaining_time, 1.0))
    
    def _get_available_connection(self) -> Optional[PooledConnection]:
        """
        Get an available connection from the pool.
        
        Returns:
            Available connection or None
        """
        while self._available_connections:
            conn = self._available_connections.pop(0)
            
            # Check if connection should be recycled
            if conn.should_recycle(self.max_age, self.max_uses):
                self._remove_connection(conn)
                continue
            
            return conn
        
        return None
    
    def _validate_connection(self, connection: PooledConnection) -> bool:
        """
        Validate a connection before use.
        
        Args:
            connection: Connection to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            is_valid = self.validator.validate(connection.connection)
            
            if not is_valid:
                # Try to repair
                is_valid = self.validator.repair(connection.connection)
            
            return is_valid
        except Exception as e:
            self.logger.debug(f"Connection validation error: {e}")
            return False
    
    def return_connection(self, connection: Any) -> None:
        """
        Return a connection to the pool.
        
        Args:
            connection: Connection to return
        """
        if self._closed:
            # If pool is closed, just close the connection
            try:
                connection.close()
            except:
                pass
            return
        
        thread_id = threading.current_thread().ident
        
        with self._lock:
            pooled_conn = self._in_use_connections.get(thread_id)
            
            if pooled_conn and pooled_conn.connection == connection:
                # Remove from in-use
                del self._in_use_connections[thread_id]
                pooled_conn.release()
                
                try:
                    # Reset connection state
                    connection.rollback()
                    
                    # Return to available pool
                    self._available_connections.append(pooled_conn)
                    
                    # Update metrics
                    self.metrics.increment('connections_released')
                    self.metrics.set('active_connections', len(self._in_use_connections))
                    self.metrics.set('idle_connections', len(self._available_connections))
                    
                    # Notify waiting threads
                    self._condition.notify()
                except Exception as e:
                    self.logger.warning(f"Error returning connection: {e}")
                    self._remove_connection(pooled_conn)
            else:
                self.logger.warning("Attempted to return unknown connection")
    
    def _remove_connection(self, connection: PooledConnection) -> None:
        """
        Remove a connection from the pool.
        
        Args:
            connection: Connection to remove
        """
        try:
            # Close the underlying connection
            connection.connection.close()
        except:
            pass
        
        # Remove from all lists
        if connection in self._connections:
            self._connections.remove(connection)
        if connection in self._available_connections:
            self._available_connections.remove(connection)
        
        self.metrics.increment('connections_destroyed')
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._lock:
            self._closed = True
            
            # Close all connections
            for conn in self._connections:
                try:
                    conn.connection.close()
                except:
                    pass
            
            self._connections.clear()
            self._available_connections.clear()
            self._in_use_connections.clear()
            
            self.logger.info("All connections closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics.
        
        Returns:
            Dictionary of pool statistics
        """
        with self._lock:
            stats = self.metrics.get_stats()
            stats.update({
                'total_connections': len(self._connections),
                'available_connections': len(self._available_connections),
                'in_use_connections': len(self._in_use_connections),
                'min_connections': self.min_connections,
                'max_connections': self.max_connections,
                'closed': self._closed
            })
            return stats
    
    def health_check(self) -> bool:
        """
        Check pool health.
        
        Returns:
            True if pool is healthy
        """
        if self._closed:
            return False
        
        try:
            # Try to get and return a connection
            conn = self.get_connection()
            self.return_connection(conn)
            
            self.metrics.set('last_health_check', datetime.now())
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self.metrics.increment('health_check_failures')
            return False
    
    def _start_maintenance_thread(self) -> None:
        """Start background thread for pool maintenance."""
        def maintenance_loop():
            while not self._closed:
                try:
                    self._perform_maintenance()
                except Exception as e:
                    self.logger.error(f"Maintenance error: {e}")
                
                # Sleep for maintenance interval
                time.sleep(30)
        
        thread = threading.Thread(target=maintenance_loop, daemon=True)
        thread.start()
    
    def _perform_maintenance(self) -> None:
        """Perform periodic pool maintenance."""
        with self._lock:
            # Remove idle connections
            now = datetime.now()
            idle_connections = [
                conn for conn in self._available_connections
                if conn.idle_seconds() > self.idle_timeout
            ]
            
            for conn in idle_connections:
                if len(self._connections) > self.min_connections:
                    self._remove_connection(conn)
                    self.logger.debug("Removed idle connection")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_all()