"""
Base exception classes for the trading platform.
Provides structured error handling with context preservation.
"""

from typing import Any, Dict, Optional
import traceback
from datetime import datetime


class TradingPlatformError(Exception):
    """
    Base exception for all trading platform errors.
    Provides structured error information and context.
    """
    
    def __init__(self, 
                 message: str,
                 error_code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None,
                 cause: Optional[Exception] = None):
        """
        Initialize trading platform error.
        
        Args:
            message: Error message
            error_code: Unique error code for tracking
            details: Additional error details
            cause: Underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()
        
        # Capture stack trace
        self.stack_trace = traceback.format_stack()
        
        # Add cause details if present
        if cause:
            self.details['cause'] = {
                'type': type(cause).__name__,
                'message': str(cause),
                'traceback': traceback.format_exception(type(cause), cause, cause.__traceback__)
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for logging/serialization.
        
        Returns:
            Dictionary representation of error
        """
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'stack_trace': self.stack_trace
        }
    
    def __str__(self) -> str:
        """String representation of error."""
        parts = [f"[{self.error_code}] {self.message}"]
        
        if self.details:
            parts.append(f"Details: {self.details}")
            
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
            
        return " | ".join(parts)


class ValidationError(TradingPlatformError):
    """Raised when data validation fails."""
    
    def __init__(self, field: str, value: Any, constraint: str, **kwargs):
        """
        Initialize validation error.
        
        Args:
            field: Field that failed validation
            value: Invalid value
            constraint: Constraint that was violated
            **kwargs: Additional arguments for base class
        """
        message = f"Validation failed for field '{field}': {constraint}"
        details = kwargs.pop('details', {})
        details.update({
            'field': field,
            'value': value,
            'constraint': constraint
        })
        
        super().__init__(
            message=message,
            error_code='VALIDATION_ERROR',
            details=details,
            **kwargs
        )


class ConfigurationError(TradingPlatformError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, config_key: str, reason: str, **kwargs):
        """
        Initialize configuration error.
        
        Args:
            config_key: Configuration key that has issue
            reason: Reason for configuration error
            **kwargs: Additional arguments for base class
        """
        message = f"Configuration error for '{config_key}': {reason}"
        details = kwargs.pop('details', {})
        details.update({
            'config_key': config_key,
            'reason': reason
        })
        
        super().__init__(
            message=message,
            error_code='CONFIG_ERROR',
            details=details,
            **kwargs
        )


class DataError(TradingPlatformError):
    """Base class for data-related errors."""
    pass


class DataNotFoundError(DataError):
    """Raised when requested data is not found."""
    
    def __init__(self, entity_type: str, identifier: Any, **kwargs):
        """
        Initialize data not found error.
        
        Args:
            entity_type: Type of entity not found
            identifier: Identifier used to search
            **kwargs: Additional arguments for base class
        """
        message = f"{entity_type} not found with identifier: {identifier}"
        details = kwargs.pop('details', {})
        details.update({
            'entity_type': entity_type,
            'identifier': identifier
        })
        
        super().__init__(
            message=message,
            error_code='DATA_NOT_FOUND',
            details=details,
            **kwargs
        )


class DataIntegrityError(DataError):
    """Raised when data integrity constraints are violated."""
    
    def __init__(self, constraint: str, data: Any, **kwargs):
        """
        Initialize data integrity error.
        
        Args:
            constraint: Integrity constraint violated
            data: Data that violated constraint
            **kwargs: Additional arguments for base class
        """
        message = f"Data integrity violation: {constraint}"
        details = kwargs.pop('details', {})
        details.update({
            'constraint': constraint,
            'data': str(data)
        })
        
        super().__init__(
            message=message,
            error_code='DATA_INTEGRITY',
            details=details,
            **kwargs
        )


class BusinessRuleError(TradingPlatformError):
    """Raised when business rules are violated."""
    
    def __init__(self, rule: str, context: Dict[str, Any], **kwargs):
        """
        Initialize business rule error.
        
        Args:
            rule: Business rule that was violated
            context: Context in which violation occurred
            **kwargs: Additional arguments for base class
        """
        message = f"Business rule violation: {rule}"
        details = kwargs.pop('details', {})
        details.update({
            'rule': rule,
            'context': context
        })
        
        super().__init__(
            message=message,
            error_code='BUSINESS_RULE',
            details=details,
            **kwargs
        )


class ExternalServiceError(TradingPlatformError):
    """Raised when external service interaction fails."""
    
    def __init__(self, service_name: str, operation: str, 
                 status_code: Optional[int] = None, **kwargs):
        """
        Initialize external service error.
        
        Args:
            service_name: Name of external service
            operation: Operation that failed
            status_code: HTTP status code if applicable
            **kwargs: Additional arguments for base class
        """
        message = f"External service error: {service_name} failed during {operation}"
        details = kwargs.pop('details', {})
        details.update({
            'service_name': service_name,
            'operation': operation,
            'status_code': status_code
        })
        
        super().__init__(
            message=message,
            error_code='EXTERNAL_SERVICE',
            details=details,
            **kwargs
        )


class RetryableError(TradingPlatformError):
    """Base class for errors that should trigger retry logic."""
    
    def __init__(self, *args, max_retries: int = 3, **kwargs):
        """
        Initialize retryable error.
        
        Args:
            max_retries: Maximum number of retries
            *args: Arguments for base class
            **kwargs: Additional arguments for base class
        """
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.details['max_retries'] = max_retries


class RateLimitError(RetryableError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, service: str, limit: int, 
                 reset_time: Optional[datetime] = None, **kwargs):
        """
        Initialize rate limit error.
        
        Args:
            service: Service that imposed rate limit
            limit: Rate limit that was exceeded
            reset_time: When rate limit resets
            **kwargs: Additional arguments for base class
        """
        message = f"Rate limit exceeded for {service}: {limit} requests"
        details = kwargs.pop('details', {})
        details.update({
            'service': service,
            'limit': limit,
            'reset_time': reset_time.isoformat() if reset_time else None
        })
        
        super().__init__(
            message=message,
            error_code='RATE_LIMIT',
            details=details,
            **kwargs
        )


class SecurityError(TradingPlatformError):
    """Base class for security-related errors."""
    
    def __init__(self, *args, log_level: str = 'CRITICAL', **kwargs):
        """
        Initialize security error.
        
        Args:
            log_level: Logging level for security errors
            *args: Arguments for base class
            **kwargs: Additional arguments for base class
        """
        super().__init__(*args, **kwargs)
        self.log_level = log_level


class AuthenticationError(SecurityError):
    """Raised when authentication fails."""
    
    def __init__(self, method: str, reason: str, **kwargs):
        """
        Initialize authentication error.
        
        Args:
            method: Authentication method that failed
            reason: Reason for failure
            **kwargs: Additional arguments for base class
        """
        message = f"Authentication failed using {method}: {reason}"
        details = kwargs.pop('details', {})
        details.update({
            'method': method,
            'reason': reason
        })
        
        super().__init__(
            message=message,
            error_code='AUTH_FAILED',
            details=details,
            **kwargs
        )


class AuthorizationError(SecurityError):
    """Raised when authorization fails."""
    
    def __init__(self, resource: str, action: str, user: Optional[str] = None, **kwargs):
        """
        Initialize authorization error.
        
        Args:
            resource: Resource being accessed
            action: Action being attempted
            user: User attempting action
            **kwargs: Additional arguments for base class
        """
        message = f"Unauthorized access to {resource} for action: {action}"
        details = kwargs.pop('details', {})
        details.update({
            'resource': resource,
            'action': action,
            'user': user
        })
        
        super().__init__(
            message=message,
            error_code='AUTH_DENIED',
            details=details,
            **kwargs
        )