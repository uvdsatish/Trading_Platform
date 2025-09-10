"""
Retry logic with exponential backoff for database operations.
"""

import time
import random
import logging
from functools import wraps
from typing import Callable, Any, Optional, Tuple, Type
from psycopg2 import OperationalError, DatabaseError

from .exceptions import (
    RetryableError,
    DatabaseTimeoutError,
    DatabaseConnectionLostError,
    DeadlockError
)


class RetryPolicy:
    """
    Defines retry behavior for database operations.
    """
    
    def __init__(self,
                 max_attempts: int = 3,
                 initial_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        """
        Initialize retry policy.
        
        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number.
        
        Args:
            attempt: Attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            delay = delay * (0.5 + random.random())
        
        return delay
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if operation should be retried.
        
        Args:
            exception: Exception that occurred
            attempt: Current attempt number (0-based)
            
        Returns:
            True if should retry, False otherwise
        """
        # Check if we've exceeded max attempts
        if attempt >= self.max_attempts - 1:
            return False
        
        # Check if exception is retryable
        return self.is_retryable_error(exception)
    
    def is_retryable_error(self, exception: Exception) -> bool:
        """
        Check if an error is retryable.
        
        Args:
            exception: Exception to check
            
        Returns:
            True if retryable, False otherwise
        """
        # Check for specific retryable exceptions
        if isinstance(exception, RetryableError):
            return True
        
        # Check for psycopg2 operational errors
        if isinstance(exception, OperationalError):
            error_message = str(exception).lower()
            retryable_messages = [
                'connection',
                'timeout',
                'could not connect',
                'connection refused',
                'connection reset',
                'broken pipe',
                'server closed the connection'
            ]
            return any(msg in error_message for msg in retryable_messages)
        
        # Check for deadlocks
        if isinstance(exception, (DeadlockError, DatabaseError)):
            error_message = str(exception).lower()
            if 'deadlock' in error_message or 'lock timeout' in error_message:
                return True
        
        return False


def with_retry(retry_policy: Optional[RetryPolicy] = None):
    """
    Decorator to add retry logic to database operations.
    
    Args:
        retry_policy: Retry policy to use (uses default if None)
        
    Returns:
        Decorated function
        
    Example:
        @with_retry(RetryPolicy(max_attempts=5))
        def fetch_data():
            # Database operation
            pass
    """
    if retry_policy is None:
        retry_policy = RetryPolicy()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(retry_policy.max_attempts):
                try:
                    # Log retry attempt if not first attempt
                    if attempt > 0:
                        retry_policy.logger.info(
                            f"Retrying {func.__name__} (attempt {attempt + 1}/{retry_policy.max_attempts})"
                        )
                    
                    # Execute the function
                    return func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry
                    if not retry_policy.should_retry(e, attempt):
                        retry_policy.logger.error(
                            f"Non-retryable error in {func.__name__}: {e}"
                        )
                        raise
                    
                    # Calculate delay
                    delay = retry_policy.calculate_delay(attempt)
                    
                    retry_policy.logger.warning(
                        f"Retryable error in {func.__name__}: {e}. "
                        f"Waiting {delay:.2f}s before retry..."
                    )
                    
                    # Wait before retry
                    time.sleep(delay)
            
            # If we've exhausted all retries, raise the last exception
            retry_policy.logger.error(
                f"All retry attempts failed for {func.__name__}"
            )
            raise last_exception
        
        return wrapper
    return decorator


class RetryableOperation:
    """
    Context manager for retryable database operations.
    """
    
    def __init__(self, 
                 operation_name: str,
                 retry_policy: Optional[RetryPolicy] = None):
        """
        Initialize retryable operation.
        
        Args:
            operation_name: Name of the operation (for logging)
            retry_policy: Retry policy to use
        """
        self.operation_name = operation_name
        self.retry_policy = retry_policy or RetryPolicy()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.attempt = 0
        self.start_time = None
    
    def __enter__(self):
        """Enter context manager."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context manager with retry logic.
        
        Returns:
            True to suppress exception if retry is needed
        """
        if exc_type is None:
            # No exception, operation succeeded
            if self.attempt > 0:
                elapsed = time.time() - self.start_time
                self.logger.info(
                    f"Operation '{self.operation_name}' succeeded after "
                    f"{self.attempt + 1} attempts in {elapsed:.2f}s"
                )
            return False
        
        # Check if we should retry
        if self.retry_policy.should_retry(exc_val, self.attempt):
            delay = self.retry_policy.calculate_delay(self.attempt)
            
            self.logger.warning(
                f"Retryable error in '{self.operation_name}': {exc_val}. "
                f"Attempt {self.attempt + 1}/{self.retry_policy.max_attempts}. "
                f"Waiting {delay:.2f}s..."
            )
            
            time.sleep(delay)
            self.attempt += 1
            
            # Suppress exception to allow retry
            return True
        
        # Don't retry, let exception propagate
        return False
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        last_exception = None
        
        for self.attempt in range(self.retry_policy.max_attempts):
            try:
                with self:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if not self.retry_policy.should_retry(e, self.attempt):
                    raise
        
        raise last_exception


class CircuitBreaker:
    """
    Circuit breaker pattern for database operations to prevent cascading failures.
    """
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery (seconds)
            expected_exception: Exception type to monitor
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == 'open':
            if self._should_attempt_reset():
                self.state = 'half_open'
                self.logger.info("Circuit breaker entering half-open state")
            else:
                raise ConnectionError(
                    f"Circuit breaker is open. Waiting for recovery timeout."
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """
        Check if we should attempt to reset the circuit.
        
        Returns:
            True if should attempt reset
        """
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == 'half_open':
            self.logger.info("Circuit breaker closing after successful call")
            self.state = 'closed'
        
        self.failure_count = 0
        self.last_failure_time = None
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.logger.warning(
                f"Circuit breaker opening after {self.failure_count} failures"
            )
            self.state = 'open'
    
    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'
        self.logger.info("Circuit breaker manually reset")
    
    def get_state(self) -> dict:
        """
        Get current circuit breaker state.
        
        Returns:
            State information
        """
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time,
            'failure_threshold': self.failure_threshold,
            'recovery_timeout': self.recovery_timeout
        }