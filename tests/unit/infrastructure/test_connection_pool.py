"""
Unit tests for database connection pool system.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import threading
import time
import psycopg2
from psycopg2 import OperationalError

from src.infrastructure.database.connection_pool import PostgreSQLConnectionPool
from src.infrastructure.database.base import PooledConnection, ConnectionValidator
from src.infrastructure.database.exceptions import (
    ConnectionPoolExhaustedError,
    ConnectionAcquisitionError,
    ConnectionPoolError
)
from src.config import DatabaseConfig


class TestPostgreSQLConnectionPool(unittest.TestCase):
    """Test cases for PostgreSQL connection pool."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock database configuration
        self.mock_config = Mock(spec=DatabaseConfig)
        self.mock_config.get_connection_string.return_value = (
            "postgresql://test:pass@localhost:5432/testdb"
        )
        self.mock_config.get_pool_config.return_value = {
            'pool_size': 2,
            'max_overflow': 5,
            'pool_timeout': 30,
            'pool_recycle': 3600,
            'pool_pre_ping': True
        }
    
    @patch('psycopg2.connect')
    def test_pool_initialization(self, mock_connect):
        """Test that pool initializes with minimum connections."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        pool = PostgreSQLConnectionPool(
            self.mock_config,
            min_connections=2,
            max_connections=5
        )
        
        # Should create 2 initial connections
        self.assertEqual(mock_connect.call_count, 2)
        self.assertEqual(len(pool._connections), 2)
        self.assertEqual(len(pool._available_connections), 2)
        
        pool.close_all()
    
    @patch('psycopg2.connect')
    def test_get_connection(self, mock_connect):
        """Test getting a connection from the pool."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        pool = PostgreSQLConnectionPool(
            self.mock_config,
            min_connections=1,
            max_connections=3
        )
        
        # Get a connection
        conn = pool.get_connection()
        
        self.assertIsNotNone(conn)
        self.assertEqual(conn, mock_conn)
        self.assertEqual(len(pool._in_use_connections), 1)
        self.assertEqual(len(pool._available_connections), 0)
        
        pool.close_all()
    
    @patch('psycopg2.connect')
    def test_return_connection(self, mock_connect):
        """Test returning a connection to the pool."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        pool = PostgreSQLConnectionPool(
            self.mock_config,
            min_connections=1,
            max_connections=3
        )
        
        # Get and return a connection
        conn = pool.get_connection()
        pool.return_connection(conn)
        
        # Connection should be back in available pool
        self.assertEqual(len(pool._in_use_connections), 0)
        self.assertEqual(len(pool._available_connections), 1)
        
        # Rollback should have been called
        mock_conn.rollback.assert_called_once()
        
        pool.close_all()
    
    @patch('psycopg2.connect')
    def test_pool_exhaustion(self, mock_connect):
        """Test pool exhaustion behavior."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        pool = PostgreSQLConnectionPool(
            self.mock_config,
            min_connections=1,
            max_connections=2,
            connection_timeout=0.5  # Short timeout for testing
        )
        
        # Get all available connections
        conn1 = pool.get_connection()
        conn2 = pool.get_connection()
        
        # Pool should be exhausted
        with self.assertRaises(ConnectionPoolExhaustedError):
            pool.get_connection()
        
        pool.close_all()
    
    @patch('psycopg2.connect')
    def test_connection_creation_on_demand(self, mock_connect):
        """Test that connections are created on demand up to max."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        pool = PostgreSQLConnectionPool(
            self.mock_config,
            min_connections=1,
            max_connections=3
        )
        
        # Initially 1 connection created
        self.assertEqual(mock_connect.call_count, 1)
        
        # Get 3 connections (should create 2 more)
        conn1 = pool.get_connection()
        conn2 = pool.get_connection()
        conn3 = pool.get_connection()
        
        # Should have created 3 total connections
        self.assertEqual(mock_connect.call_count, 3)
        self.assertEqual(len(pool._connections), 3)
        
        pool.close_all()
    
    @patch('psycopg2.connect')
    def test_connection_validation(self, mock_connect):
        """Test connection validation before checkout."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        pool = PostgreSQLConnectionPool(
            self.mock_config,
            min_connections=1,
            max_connections=3,
            pre_ping=True
        )
        
        # Get a connection (validation should occur)
        conn = pool.get_connection()
        
        # Validation query should have been executed
        mock_cursor.execute.assert_any_call("SELECT 1")
        
        pool.close_all()
    
    @patch('psycopg2.connect')
    def test_connection_recycling(self, mock_connect):
        """Test that old connections are recycled."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        pool = PostgreSQLConnectionPool(
            self.mock_config,
            min_connections=1,
            max_connections=3,
            max_age=0.1,  # Very short for testing
            max_uses=1
        )
        
        # Get and return a connection
        conn = pool.get_connection()
        pool.return_connection(conn)
        
        # Wait for connection to age
        time.sleep(0.2)
        
        # Get connection again - should create new one due to age
        conn = pool.get_connection()
        
        # Should have closed the old connection
        mock_conn.close.assert_called()
        
        pool.close_all()
    
    @patch('psycopg2.connect')
    def test_concurrent_access(self, mock_connect):
        """Test thread-safe concurrent access to pool."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        pool = PostgreSQLConnectionPool(
            self.mock_config,
            min_connections=2,
            max_connections=5
        )
        
        results = []
        errors = []
        
        def get_and_return_connection():
            try:
                conn = pool.get_connection()
                time.sleep(0.01)  # Simulate some work
                pool.return_connection(conn)
                results.append("success")
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=get_and_return_connection)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # All operations should succeed
        self.assertEqual(len(results), 10)
        self.assertEqual(len(errors), 0)
        
        pool.close_all()
    
    @patch('psycopg2.connect')
    def test_health_check(self, mock_connect):
        """Test pool health check."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        pool = PostgreSQLConnectionPool(
            self.mock_config,
            min_connections=1,
            max_connections=3
        )
        
        # Health check should succeed
        self.assertTrue(pool.health_check())
        
        # Close pool and check again
        pool.close_all()
        self.assertFalse(pool.health_check())
    
    @patch('psycopg2.connect')
    def test_get_stats(self, mock_connect):
        """Test getting pool statistics."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        pool = PostgreSQLConnectionPool(
            self.mock_config,
            min_connections=2,
            max_connections=5
        )
        
        # Get initial stats
        stats = pool.get_stats()
        
        self.assertEqual(stats['total_connections'], 2)
        self.assertEqual(stats['available_connections'], 2)
        self.assertEqual(stats['in_use_connections'], 0)
        self.assertEqual(stats['connections_created'], 2)
        
        # Get a connection and check stats
        conn = pool.get_connection()
        stats = pool.get_stats()
        
        self.assertEqual(stats['available_connections'], 1)
        self.assertEqual(stats['in_use_connections'], 1)
        self.assertEqual(stats['connections_acquired'], 1)
        
        pool.close_all()
    
    @patch('psycopg2.connect')
    def test_connection_failure_handling(self, mock_connect):
        """Test handling of connection creation failures."""
        # First connection succeeds, second fails
        mock_conn = Mock()
        mock_connect.side_effect = [
            mock_conn,
            OperationalError("Connection failed")
        ]
        
        with self.assertRaises(ConnectionPoolError):
            pool = PostgreSQLConnectionPool(
                self.mock_config,
                min_connections=2,  # Requires 2 connections
                max_connections=5
            )


class TestConnectionValidator(unittest.TestCase):
    """Test cases for connection validator."""
    
    def test_validate_success(self):
        """Test successful connection validation."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor
        
        validator = ConnectionValidator()
        result = validator.validate(mock_conn)
        
        self.assertTrue(result)
        mock_cursor.execute.assert_any_call("SELECT 1")
    
    def test_validate_failure(self):
        """Test failed connection validation."""
        mock_conn = Mock()
        mock_conn.cursor.side_effect = Exception("Connection dead")
        
        validator = ConnectionValidator()
        result = validator.validate(mock_conn)
        
        self.assertFalse(result)
    
    def test_repair_connection(self):
        """Test connection repair."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor
        
        validator = ConnectionValidator()
        result = validator.repair(mock_conn)
        
        self.assertTrue(result)
        mock_conn.rollback.assert_called_once()


class TestPooledConnection(unittest.TestCase):
    """Test cases for PooledConnection wrapper."""
    
    def test_acquire_release(self):
        """Test acquiring and releasing a pooled connection."""
        mock_conn = Mock()
        pooled = PooledConnection(mock_conn, "test_pool")
        
        # Initially not in use
        self.assertFalse(pooled.in_use)
        self.assertIsNone(pooled.thread_id)
        
        # Acquire connection
        pooled.acquire(12345)
        self.assertTrue(pooled.in_use)
        self.assertEqual(pooled.thread_id, 12345)
        self.assertEqual(pooled.use_count, 1)
        
        # Release connection
        pooled.release()
        self.assertFalse(pooled.in_use)
        self.assertIsNone(pooled.thread_id)
    
    def test_should_recycle(self):
        """Test connection recycling logic."""
        mock_conn = Mock()
        pooled = PooledConnection(mock_conn, "test_pool")
        
        # Initially should not recycle
        self.assertFalse(pooled.should_recycle(max_age=3600, max_uses=1000))
        
        # Simulate many uses
        for _ in range(1001):
            pooled.acquire()
            pooled.release()
        
        # Should recycle due to use count
        self.assertTrue(pooled.should_recycle(max_age=3600, max_uses=1000))
    
    def test_age_calculation(self):
        """Test connection age calculation."""
        mock_conn = Mock()
        pooled = PooledConnection(mock_conn, "test_pool")
        
        # Age should be close to 0
        age = pooled.age_seconds()
        self.assertLess(age, 1)
        
        # Simulate aging
        time.sleep(0.1)
        age = pooled.age_seconds()
        self.assertGreater(age, 0.09)


if __name__ == '__main__':
    unittest.main()