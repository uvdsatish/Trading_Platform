"""
Domain-specific exceptions for different areas of the trading platform.
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from decimal import Decimal

from .base import TradingPlatformError, DataError, BusinessRuleError


# Market Data Exceptions
class MarketDataError(DataError):
    """Base class for market data related errors."""
    pass


class TickerNotFoundError(MarketDataError):
    """Raised when a ticker symbol is not found."""
    
    def __init__(self, ticker: str, exchange: Optional[str] = None, **kwargs):
        """
        Initialize ticker not found error.
        
        Args:
            ticker: Ticker symbol
            exchange: Exchange where ticker was searched
            **kwargs: Additional arguments
        """
        message = f"Ticker '{ticker}' not found"
        if exchange:
            message += f" on exchange '{exchange}'"
            
        details = kwargs.pop('details', {})
        details.update({
            'ticker': ticker,
            'exchange': exchange
        })
        
        super().__init__(
            entity_type='Ticker',
            identifier=ticker,
            message=message,
            error_code='TICKER_NOT_FOUND',
            details=details,
            **kwargs
        )


class MarketDataUnavailableError(MarketDataError):
    """Raised when market data is temporarily unavailable."""
    
    def __init__(self, data_type: str, reason: str, **kwargs):
        """
        Initialize market data unavailable error.
        
        Args:
            data_type: Type of data unavailable
            reason: Reason for unavailability
            **kwargs: Additional arguments
        """
        message = f"Market data unavailable for {data_type}: {reason}"
        details = kwargs.pop('details', {})
        details.update({
            'data_type': data_type,
            'reason': reason
        })
        
        super().__init__(
            message=message,
            error_code='MARKET_DATA_UNAVAILABLE',
            details=details,
            **kwargs
        )


class StaleDataError(MarketDataError):
    """Raised when data is stale beyond acceptable threshold."""
    
    def __init__(self, data_type: str, last_update: datetime, 
                 threshold_minutes: int, **kwargs):
        """
        Initialize stale data error.
        
        Args:
            data_type: Type of stale data
            last_update: Last update timestamp
            threshold_minutes: Staleness threshold in minutes
            **kwargs: Additional arguments
        """
        age_minutes = (datetime.utcnow() - last_update).total_seconds() / 60
        message = f"Data for {data_type} is stale: {age_minutes:.1f} minutes old (threshold: {threshold_minutes})"
        
        details = kwargs.pop('details', {})
        details.update({
            'data_type': data_type,
            'last_update': last_update.isoformat(),
            'age_minutes': age_minutes,
            'threshold_minutes': threshold_minutes
        })
        
        super().__init__(
            message=message,
            error_code='STALE_DATA',
            details=details,
            **kwargs
        )


# Trading Exceptions
class TradingError(BusinessRuleError):
    """Base class for trading related errors."""
    pass


class InsufficientFundsError(TradingError):
    """Raised when account has insufficient funds for trade."""
    
    def __init__(self, required: Decimal, available: Decimal, 
                 account_id: str, **kwargs):
        """
        Initialize insufficient funds error.
        
        Args:
            required: Required amount
            available: Available amount
            account_id: Account identifier
            **kwargs: Additional arguments
        """
        message = f"Insufficient funds: required ${required}, available ${available}"
        context = {
            'required': float(required),
            'available': float(available),
            'account_id': account_id,
            'shortfall': float(required - available)
        }
        
        super().__init__(
            rule='Sufficient funds required for trade',
            context=context,
            message=message,
            error_code='INSUFFICIENT_FUNDS',
            **kwargs
        )


class PositionLimitExceededError(TradingError):
    """Raised when position limits are exceeded."""
    
    def __init__(self, ticker: str, current_position: int, 
                 requested: int, limit: int, **kwargs):
        """
        Initialize position limit exceeded error.
        
        Args:
            ticker: Ticker symbol
            current_position: Current position size
            requested: Requested additional position
            limit: Position limit
            **kwargs: Additional arguments
        """
        total = current_position + requested
        message = f"Position limit exceeded for {ticker}: current={current_position}, requested={requested}, limit={limit}"
        
        context = {
            'ticker': ticker,
            'current_position': current_position,
            'requested': requested,
            'total_would_be': total,
            'limit': limit
        }
        
        super().__init__(
            rule='Position size must not exceed limit',
            context=context,
            message=message,
            error_code='POSITION_LIMIT_EXCEEDED',
            **kwargs
        )


class InvalidOrderError(TradingError):
    """Raised when order parameters are invalid."""
    
    def __init__(self, order_type: str, reason: str, 
                 order_details: Dict[str, Any], **kwargs):
        """
        Initialize invalid order error.
        
        Args:
            order_type: Type of order
            reason: Reason order is invalid
            order_details: Order details
            **kwargs: Additional arguments
        """
        message = f"Invalid {order_type} order: {reason}"
        
        context = {
            'order_type': order_type,
            'reason': reason,
            'order_details': order_details
        }
        
        super().__init__(
            rule='Order parameters must be valid',
            context=context,
            message=message,
            error_code='INVALID_ORDER',
            **kwargs
        )


class MarketClosedError(TradingError):
    """Raised when attempting to trade while market is closed."""
    
    def __init__(self, market: str, current_time: datetime, 
                 next_open: Optional[datetime] = None, **kwargs):
        """
        Initialize market closed error.
        
        Args:
            market: Market name
            current_time: Current time
            next_open: Next market open time
            **kwargs: Additional arguments
        """
        message = f"Market '{market}' is closed"
        if next_open:
            message += f", opens at {next_open.isoformat()}"
            
        context = {
            'market': market,
            'current_time': current_time.isoformat(),
            'next_open': next_open.isoformat() if next_open else None
        }
        
        super().__init__(
            rule='Trading only allowed during market hours',
            context=context,
            message=message,
            error_code='MARKET_CLOSED',
            **kwargs
        )


# Technical Analysis Exceptions
class TechnicalAnalysisError(TradingPlatformError):
    """Base class for technical analysis errors."""
    pass


class InsufficientDataError(TechnicalAnalysisError):
    """Raised when insufficient data for calculation."""
    
    def __init__(self, indicator: str, required_periods: int, 
                 available_periods: int, **kwargs):
        """
        Initialize insufficient data error.
        
        Args:
            indicator: Indicator name
            required_periods: Required number of periods
            available_periods: Available number of periods
            **kwargs: Additional arguments
        """
        message = (f"Insufficient data for {indicator}: "
                  f"requires {required_periods} periods, have {available_periods}")
        
        details = kwargs.pop('details', {})
        details.update({
            'indicator': indicator,
            'required_periods': required_periods,
            'available_periods': available_periods,
            'shortfall': required_periods - available_periods
        })
        
        super().__init__(
            message=message,
            error_code='INSUFFICIENT_DATA',
            details=details,
            **kwargs
        )


class InvalidIndicatorParametersError(TechnicalAnalysisError):
    """Raised when indicator parameters are invalid."""
    
    def __init__(self, indicator: str, parameters: Dict[str, Any], 
                 reason: str, **kwargs):
        """
        Initialize invalid indicator parameters error.
        
        Args:
            indicator: Indicator name
            parameters: Invalid parameters
            reason: Reason parameters are invalid
            **kwargs: Additional arguments
        """
        message = f"Invalid parameters for {indicator}: {reason}"
        
        details = kwargs.pop('details', {})
        details.update({
            'indicator': indicator,
            'parameters': parameters,
            'reason': reason
        })
        
        super().__init__(
            message=message,
            error_code='INVALID_INDICATOR_PARAMS',
            details=details,
            **kwargs
        )


# Backtesting Exceptions
class BacktestingError(TradingPlatformError):
    """Base class for backtesting errors."""
    pass


class BacktestDataError(BacktestingError):
    """Raised when backtest data has issues."""
    
    def __init__(self, issue: str, start_date: datetime, 
                 end_date: datetime, tickers: List[str], **kwargs):
        """
        Initialize backtest data error.
        
        Args:
            issue: Description of data issue
            start_date: Backtest start date
            end_date: Backtest end date
            tickers: Tickers involved
            **kwargs: Additional arguments
        """
        message = f"Backtest data issue: {issue}"
        
        details = kwargs.pop('details', {})
        details.update({
            'issue': issue,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'tickers': tickers,
            'date_range_days': (end_date - start_date).days
        })
        
        super().__init__(
            message=message,
            error_code='BACKTEST_DATA_ERROR',
            details=details,
            **kwargs
        )


class StrategyExecutionError(BacktestingError):
    """Raised when strategy execution fails during backtest."""
    
    def __init__(self, strategy_name: str, timestamp: datetime, 
                 reason: str, **kwargs):
        """
        Initialize strategy execution error.
        
        Args:
            strategy_name: Name of strategy
            timestamp: When error occurred
            reason: Reason for failure
            **kwargs: Additional arguments
        """
        message = f"Strategy '{strategy_name}' failed at {timestamp}: {reason}"
        
        details = kwargs.pop('details', {})
        details.update({
            'strategy_name': strategy_name,
            'timestamp': timestamp.isoformat(),
            'reason': reason
        })
        
        super().__init__(
            message=message,
            error_code='STRATEGY_EXECUTION_ERROR',
            details=details,
            **kwargs
        )


# Data Provider Exceptions
class DataProviderError(TradingPlatformError):
    """Base class for data provider errors."""
    pass


class IQFeedError(DataProviderError):
    """Raised when IQFeed operations fail."""
    
    def __init__(self, operation: str, error_message: str, **kwargs):
        """
        Initialize IQFeed error.
        
        Args:
            operation: Operation that failed
            error_message: IQFeed error message
            **kwargs: Additional arguments
        """
        message = f"IQFeed error during {operation}: {error_message}"
        
        details = kwargs.pop('details', {})
        details.update({
            'provider': 'IQFeed',
            'operation': operation,
            'error_message': error_message
        })
        
        super().__init__(
            message=message,
            error_code='IQFEED_ERROR',
            details=details,
            **kwargs
        )


class AlphaVantageError(DataProviderError):
    """Raised when Alpha Vantage API operations fail."""
    
    def __init__(self, endpoint: str, status_code: int, 
                 response: str, **kwargs):
        """
        Initialize Alpha Vantage error.
        
        Args:
            endpoint: API endpoint
            status_code: HTTP status code
            response: API response
            **kwargs: Additional arguments
        """
        message = f"Alpha Vantage API error: {status_code} from {endpoint}"
        
        details = kwargs.pop('details', {})
        details.update({
            'provider': 'AlphaVantage',
            'endpoint': endpoint,
            'status_code': status_code,
            'response': response
        })
        
        super().__init__(
            message=message,
            error_code='ALPHAVANTAGE_ERROR',
            details=details,
            **kwargs
        )