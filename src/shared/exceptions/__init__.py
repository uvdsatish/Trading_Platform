"""
Exception handling for the trading platform.

Provides structured exception hierarchy with context preservation
and automatic logging integration.

Usage Examples:
    # Raising domain exceptions
    from src.shared.exceptions import TickerNotFoundError, InsufficientFundsError
    
    raise TickerNotFoundError('INVALID', exchange='NYSE')
    
    # Handling with context
    from src.shared.exceptions import TradingPlatformError
    
    try:
        # Your code
    except TradingPlatformError as e:
        logger.error("Operation failed", exception=e)
        print(e.to_dict())  # Get structured error info
"""

from .base import (
    TradingPlatformError,
    ValidationError,
    ConfigurationError,
    DataError,
    DataNotFoundError,
    DataIntegrityError,
    BusinessRuleError,
    ExternalServiceError,
    RetryableError,
    RateLimitError,
    SecurityError,
    AuthenticationError,
    AuthorizationError
)

from .domain import (
    # Market Data
    MarketDataError,
    TickerNotFoundError,
    MarketDataUnavailableError,
    StaleDataError,
    
    # Trading
    TradingError,
    InsufficientFundsError,
    PositionLimitExceededError,
    InvalidOrderError,
    MarketClosedError,
    
    # Technical Analysis
    TechnicalAnalysisError,
    InsufficientDataError,
    InvalidIndicatorParametersError,
    
    # Backtesting
    BacktestingError,
    BacktestDataError,
    StrategyExecutionError,
    
    # Data Providers
    DataProviderError,
    IQFeedError,
    AlphaVantageError
)

__all__ = [
    # Base exceptions
    'TradingPlatformError',
    'ValidationError',
    'ConfigurationError',
    'DataError',
    'DataNotFoundError',
    'DataIntegrityError',
    'BusinessRuleError',
    'ExternalServiceError',
    'RetryableError',
    'RateLimitError',
    'SecurityError',
    'AuthenticationError',
    'AuthorizationError',
    
    # Market Data exceptions
    'MarketDataError',
    'TickerNotFoundError',
    'MarketDataUnavailableError',
    'StaleDataError',
    
    # Trading exceptions
    'TradingError',
    'InsufficientFundsError',
    'PositionLimitExceededError',
    'InvalidOrderError',
    'MarketClosedError',
    
    # Technical Analysis exceptions
    'TechnicalAnalysisError',
    'InsufficientDataError',
    'InvalidIndicatorParametersError',
    
    # Backtesting exceptions
    'BacktestingError',
    'BacktestDataError',
    'StrategyExecutionError',
    
    # Data Provider exceptions
    'DataProviderError',
    'IQFeedError',
    'AlphaVantageError',
]