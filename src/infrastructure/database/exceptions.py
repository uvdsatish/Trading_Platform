"""
Database-specific exceptions for the trading platform.
"""


class DatabaseError(Exception):
    """Base exception for all database-related errors."""
    pass


class ConnectionPoolError(DatabaseError):
    """Exception raised for connection pool errors."""
    pass


class ConnectionPoolExhaustedError(ConnectionPoolError):
    """Exception raised when no connections are available in the pool."""
    pass


class ConnectionAcquisitionError(ConnectionPoolError):
    """Exception raised when unable to acquire a connection."""
    pass


class TransactionError(DatabaseError):
    """Exception raised for transaction-related errors."""
    pass


class DeadlockError(TransactionError):
    """Exception raised when a deadlock is detected."""
    pass


class ConnectionValidationError(DatabaseError):
    """Exception raised when connection validation fails."""
    pass


class DatabaseConfigurationError(DatabaseError):
    """Exception raised for database configuration errors."""
    pass


class RetryableError(DatabaseError):
    """Base exception for errors that should trigger a retry."""
    pass


class DatabaseTimeoutError(RetryableError):
    """Exception raised when a database operation times out."""
    pass


class DatabaseConnectionLostError(RetryableError):
    """Exception raised when database connection is lost."""
    pass