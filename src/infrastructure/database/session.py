"""
Database session factory and domain-specific connection management.
"""

from typing import Dict, Optional, Any
from contextlib import contextmanager
import logging
import threading

from src.config import get_config_service, DatabaseConfig
from .connection_pool import PostgreSQLConnectionPool
from .transaction import TransactionManager
from .base import BaseConnectionManager


class DatabaseSessionFactory:
    """
    Factory for creating database sessions for different domains.
    Implements singleton pattern for connection pools per domain.
    """
    
    _instances: Dict[str, 'DatabaseSessionFactory'] = {}
    _lock = threading.Lock()
    
    def __new__(cls, domain: str = 'primary'):
        """
        Create or return existing instance for domain.
        
        Args:
            domain: Database domain name
            
        Returns:
            DatabaseSessionFactory instance
        """
        with cls._lock:
            if domain not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[domain] = instance
            return cls._instances[domain]
    
    def __init__(self, domain: str = 'primary'):
        """
        Initialize session factory for a specific domain.
        
        Args:
            domain: Database domain (primary, technical_analysis, internals, etc.)
        """
        # Skip re-initialization for singleton
        if hasattr(self, '_initialized'):
            return
        
        self.domain = domain
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{domain}")
        
        # Get configuration
        config_service = get_config_service()
        self.db_config = config_service.get_database_config(domain)
        
        # Create connection pool
        self.pool = PostgreSQLConnectionPool(self.db_config)
        
        # Create managers
        self.connection_manager = BaseConnectionManager(self.pool)
        self.transaction_manager = TransactionManager(self.connection_manager)
        
        self._initialized = True
        
        self.logger.info(f"Session factory initialized for domain: {domain}")
    
    @contextmanager
    def get_session(self):
        """
        Get a database session with automatic resource management.
        
        Yields:
            Database session
            
        Example:
            factory = DatabaseSessionFactory('technical_analysis')
            with factory.get_session() as session:
                cursor = session.cursor()
                cursor.execute("SELECT * FROM indicators")
        """
        with self.connection_manager.get_connection() as conn:
            session = DatabaseSession(conn, self.domain)
            try:
                yield session
            finally:
                session.close()
    
    @contextmanager
    def transaction(self, **kwargs):
        """
        Create a transactional session.
        
        Args:
            **kwargs: Transaction parameters (isolation_level, read_only, etc.)
            
        Yields:
            Transactional session
        """
        with self.transaction_manager.transaction(**kwargs) as tx:
            session = TransactionalSession(tx, self.domain)
            try:
                yield session
            finally:
                session.close()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Returns:
            Pool statistics
        """
        return self.pool.get_stats()
    
    def health_check(self) -> bool:
        """
        Check if the database connection is healthy.
        
        Returns:
            True if healthy
        """
        return self.pool.health_check()
    
    def close(self):
        """Close all connections in the pool."""
        self.pool.close_all()
        self.logger.info(f"Session factory closed for domain: {self.domain}")
    
    @classmethod
    def close_all(cls):
        """Close all session factories."""
        with cls._lock:
            for domain, factory in cls._instances.items():
                factory.close()
            cls._instances.clear()


class DatabaseSession:
    """
    Represents a database session with convenience methods.
    """
    
    def __init__(self, connection: Any, domain: str):
        """
        Initialize database session.
        
        Args:
            connection: Database connection
            domain: Domain name
        """
        self.connection = connection
        self.domain = domain
        self._cursors = []
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{domain}")
    
    def cursor(self, **kwargs) -> Any:
        """
        Create a cursor.
        
        Args:
            **kwargs: Cursor parameters
            
        Returns:
            Database cursor
        """
        cursor = self.connection.cursor(**kwargs)
        self._cursors.append(cursor)
        return cursor
    
    def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """
        Execute a query and return cursor.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Cursor with results
        """
        cursor = self.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor
        except Exception as e:
            cursor.close()
            self.logger.error(f"Query execution failed: {e}")
            raise
    
    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[tuple]:
        """
        Execute query and fetch one row.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Single row or None
        """
        cursor = self.execute(query, params)
        try:
            return cursor.fetchone()
        finally:
            cursor.close()
    
    def fetch_all(self, query: str, params: Optional[tuple] = None) -> list:
        """
        Execute query and fetch all rows.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            List of rows
        """
        cursor = self.execute(query, params)
        try:
            return cursor.fetchall()
        finally:
            cursor.close()
    
    def fetch_many(self, query: str, size: int, params: Optional[tuple] = None) -> list:
        """
        Execute query and fetch specified number of rows.
        
        Args:
            query: SQL query
            size: Number of rows to fetch
            params: Query parameters
            
        Returns:
            List of rows
        """
        cursor = self.execute(query, params)
        try:
            return cursor.fetchmany(size)
        finally:
            cursor.close()
    
    def commit(self):
        """Commit the current transaction."""
        self.connection.commit()
    
    def rollback(self):
        """Rollback the current transaction."""
        self.connection.rollback()
    
    def close(self):
        """Close all cursors."""
        for cursor in self._cursors:
            try:
                cursor.close()
            except:
                pass
        self._cursors.clear()


class TransactionalSession(DatabaseSession):
    """
    Database session with transaction context.
    """
    
    def __init__(self, transaction_context, domain: str):
        """
        Initialize transactional session.
        
        Args:
            transaction_context: Transaction context
            domain: Domain name
        """
        super().__init__(transaction_context.connection, domain)
        self.transaction_context = transaction_context
    
    def savepoint(self, name: str):
        """
        Create a savepoint.
        
        Args:
            name: Savepoint name
        """
        self.execute(f"SAVEPOINT {name}")
    
    def release_savepoint(self, name: str):
        """
        Release a savepoint.
        
        Args:
            name: Savepoint name
        """
        self.execute(f"RELEASE SAVEPOINT {name}")
    
    def rollback_to_savepoint(self, name: str):
        """
        Rollback to a savepoint.
        
        Args:
            name: Savepoint name
        """
        self.execute(f"ROLLBACK TO SAVEPOINT {name}")


class DomainConnectionManager:
    """
    Manages connections for specific database domains.
    """
    
    # Domain-specific session factories
    _factories: Dict[str, DatabaseSessionFactory] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_session_factory(cls, domain: str) -> DatabaseSessionFactory:
        """
        Get or create session factory for domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Session factory
        """
        with cls._lock:
            if domain not in cls._factories:
                cls._factories[domain] = DatabaseSessionFactory(domain)
            return cls._factories[domain]
    
    @classmethod
    @contextmanager
    def get_session(cls, domain: str):
        """
        Get a session for the specified domain.
        
        Args:
            domain: Domain name
            
        Yields:
            Database session
        """
        factory = cls.get_session_factory(domain)
        with factory.get_session() as session:
            yield session
    
    @classmethod
    @contextmanager  
    def transaction(cls, domain: str, **kwargs):
        """
        Get a transactional session for the specified domain.
        
        Args:
            domain: Domain name
            **kwargs: Transaction parameters
            
        Yields:
            Transactional session
        """
        factory = cls.get_session_factory(domain)
        with factory.transaction(**kwargs) as session:
            yield session
    
    @classmethod
    def get_stats(cls, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Args:
            domain: Specific domain or None for all
            
        Returns:
            Statistics dictionary
        """
        if domain:
            factory = cls.get_session_factory(domain)
            return factory.get_pool_stats()
        else:
            stats = {}
            with cls._lock:
                for domain, factory in cls._factories.items():
                    stats[domain] = factory.get_pool_stats()
            return stats
    
    @classmethod
    def close_all(cls):
        """Close all domain connections."""
        with cls._lock:
            for factory in cls._factories.values():
                factory.close()
            cls._factories.clear()


# Convenience functions for common domains
def get_technical_session():
    """Get session for technical analysis database."""
    return DomainConnectionManager.get_session('technical_analysis')

def get_internals_session():
    """Get session for market internals database."""
    return DomainConnectionManager.get_session('internals')

def get_fundamentals_session():
    """Get session for fundamentals database."""
    return DomainConnectionManager.get_session('fundamentals')

def get_macro_session():
    """Get session for macroeconomic database."""
    return DomainConnectionManager.get_session('macro')

def get_plurality_session():
    """Get session for Plurality/WAMRS database."""
    return DomainConnectionManager.get_session('plurality')

def get_backtesting_session():
    """Get session for backtesting database."""
    return DomainConnectionManager.get_session('backtesting')