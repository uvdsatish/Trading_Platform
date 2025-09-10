"""
Database infrastructure module for the trading platform.

Provides connection pooling, transaction management, and domain-specific
database access with automatic resource management and retry logic.

Usage Examples:
    # Simple query with automatic connection management
    from src.infrastructure.database import get_technical_session
    
    with get_technical_session() as session:
        results = session.fetch_all("SELECT * FROM indicators WHERE ticker = %s", ('AAPL',))
    
    # Transaction with automatic commit/rollback
    from src.infrastructure.database import DomainConnectionManager
    
    with DomainConnectionManager.transaction('technical_analysis') as session:
        session.execute("INSERT INTO indicators VALUES (%s, %s)", (date, value))
        # Automatically commits on success, rolls back on exception
    
    # Direct pool access for advanced usage
    from src.infrastructure.database import get_connection_pool
    
    pool = get_connection_pool('internals')
    with pool.get_connection() as conn:
        # Use connection directly
        pass
"""

from .base import (
    IConnectionPool,
    BaseConnectionManager,
    ConnectionPoolMetrics,
    ConnectionValidator,
    PooledConnection
)

from .connection_pool import PostgreSQLConnectionPool

from .transaction import (
    TransactionManager,
    TransactionContext,
    IsolationLevel,
    BatchTransaction,
    BatchInserter
)

from .session import (
    DatabaseSessionFactory,
    DatabaseSession,
    TransactionalSession,
    DomainConnectionManager,
    get_technical_session,
    get_internals_session,
    get_fundamentals_session,
    get_macro_session,
    get_plurality_session,
    get_backtesting_session
)

from .retry import (
    RetryPolicy,
    with_retry,
    RetryableOperation,
    CircuitBreaker
)

from .exceptions import (
    DatabaseError,
    ConnectionPoolError,
    ConnectionPoolExhaustedError,
    ConnectionAcquisitionError,
    TransactionError,
    DeadlockError,
    ConnectionValidationError,
    DatabaseConfigurationError,
    RetryableError,
    DatabaseTimeoutError,
    DatabaseConnectionLostError
)


# Convenience function to get connection pool for a domain
def get_connection_pool(domain: str = 'primary') -> PostgreSQLConnectionPool:
    """
    Get connection pool for a specific domain.
    
    Args:
        domain: Database domain name
        
    Returns:
        Connection pool instance
    """
    factory = DatabaseSessionFactory(domain)
    return factory.pool


# Initialize default pools on import (optional)
def initialize_pools(*domains: str) -> None:
    """
    Pre-initialize connection pools for specified domains.
    
    Args:
        *domains: Domain names to initialize
        
    Example:
        initialize_pools('technical_analysis', 'internals', 'fundamentals')
    """
    for domain in domains:
        DatabaseSessionFactory(domain)


# Cleanup function for application shutdown
def shutdown_all_pools() -> None:
    """
    Shutdown all connection pools gracefully.
    Should be called on application shutdown.
    """
    DatabaseSessionFactory.close_all()
    DomainConnectionManager.close_all()


__all__ = [
    # Base classes
    'IConnectionPool',
    'BaseConnectionManager',
    'ConnectionPoolMetrics',
    'ConnectionValidator',
    'PooledConnection',
    
    # Connection pool
    'PostgreSQLConnectionPool',
    'get_connection_pool',
    
    # Transaction management
    'TransactionManager',
    'TransactionContext',
    'IsolationLevel',
    'BatchTransaction',
    'BatchInserter',
    
    # Session management
    'DatabaseSessionFactory',
    'DatabaseSession',
    'TransactionalSession',
    'DomainConnectionManager',
    
    # Convenience functions
    'get_technical_session',
    'get_internals_session',
    'get_fundamentals_session',
    'get_macro_session',
    'get_plurality_session',
    'get_backtesting_session',
    
    # Retry logic
    'RetryPolicy',
    'with_retry',
    'RetryableOperation',
    'CircuitBreaker',
    
    # Exceptions
    'DatabaseError',
    'ConnectionPoolError',
    'ConnectionPoolExhaustedError',
    'ConnectionAcquisitionError',
    'TransactionError',
    'DeadlockError',
    'ConnectionValidationError',
    'DatabaseConfigurationError',
    'RetryableError',
    'DatabaseTimeoutError',
    'DatabaseConnectionLostError',
    
    # Utility functions
    'initialize_pools',
    'shutdown_all_pools',
]