"""
Unit tests for the exception handling system.
"""

import unittest
from datetime import datetime
from decimal import Decimal

from src.shared.exceptions import (
    TradingPlatformError,
    ValidationError,
    ConfigurationError,
    DataNotFoundError,
    DataIntegrityError,
    BusinessRuleError,
    ExternalServiceError,
    RetryableError,
    RateLimitError,
    SecurityError,
    AuthenticationError,
    AuthorizationError,
    # Domain exceptions
    TickerNotFoundError,
    MarketDataUnavailableError,
    StaleDataError,
    InsufficientFundsError,
    PositionLimitExceededError,
    InvalidOrderError,
    MarketClosedError,
    InsufficientDataError,
    InvalidIndicatorParametersError,
    BacktestDataError,
    StrategyExecutionError,
    IQFeedError,
    AlphaVantageError
)


class TestTradingPlatformError(unittest.TestCase):
    """Test cases for base TradingPlatformError."""
    
    def test_basic_error_creation(self):
        """Test basic error creation and properties."""
        error = TradingPlatformError(
            message="Test error message",
            error_code="TEST_ERROR",
            details={'key': 'value'}
        )
        
        self.assertEqual(str(error), "[TEST_ERROR] Test error message | Details: {'key': 'value'}")
        self.assertEqual(error.message, "Test error message")
        self.assertEqual(error.error_code, "TEST_ERROR")
        self.assertEqual(error.details, {'key': 'value'})
        self.assertIsInstance(error.timestamp, datetime)
        self.assertIsNotNone(error.stack_trace)
    
    def test_error_with_cause(self):
        """Test error creation with underlying cause."""
        original_error = ValueError("Original error")
        
        error = TradingPlatformError(
            message="Wrapper error",
            error_code="WRAPPER_ERROR",
            cause=original_error
        )
        
        self.assertEqual(error.cause, original_error)
        self.assertIn('cause', error.details)
        self.assertEqual(error.details['cause']['type'], 'ValueError')
        self.assertEqual(error.details['cause']['message'], 'Original error')
        self.assertIn('traceback', error.details['cause'])
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        error = TradingPlatformError(
            message="Test error",
            error_code="TEST_CODE",
            details={'extra': 'info'}
        )
        
        error_dict = error.to_dict()
        
        self.assertEqual(error_dict['error_type'], 'TradingPlatformError')
        self.assertEqual(error_dict['error_code'], 'TEST_CODE')
        self.assertEqual(error_dict['message'], 'Test error')
        self.assertEqual(error_dict['details'], {'extra': 'info'})
        self.assertIn('timestamp', error_dict)
        self.assertIn('stack_trace', error_dict)
    
    def test_default_error_code(self):
        """Test that error code defaults to class name."""
        error = TradingPlatformError("Test message")
        self.assertEqual(error.error_code, "TradingPlatformError")


class TestValidationError(unittest.TestCase):
    """Test cases for ValidationError."""
    
    def test_validation_error_creation(self):
        """Test validation error creation."""
        error = ValidationError(
            field="price",
            value=-100,
            constraint="must be positive"
        )
        
        self.assertIn("Validation failed for field 'price'", error.message)
        self.assertEqual(error.error_code, "VALIDATION_ERROR")
        self.assertEqual(error.details['field'], 'price')
        self.assertEqual(error.details['value'], -100)
        self.assertEqual(error.details['constraint'], 'must be positive')


class TestConfigurationError(unittest.TestCase):
    """Test cases for ConfigurationError."""
    
    def test_configuration_error_creation(self):
        """Test configuration error creation."""
        error = ConfigurationError(
            config_key="database.host",
            reason="missing required value"
        )
        
        self.assertIn("Configuration error for 'database.host'", error.message)
        self.assertEqual(error.error_code, "CONFIG_ERROR")
        self.assertEqual(error.details['config_key'], 'database.host')
        self.assertEqual(error.details['reason'], 'missing required value')


class TestDataErrors(unittest.TestCase):
    """Test cases for data-related errors."""
    
    def test_data_not_found_error(self):
        """Test DataNotFoundError."""
        error = DataNotFoundError(
            entity_type="Ticker",
            identifier="INVALID"
        )
        
        self.assertIn("Ticker not found with identifier: INVALID", error.message)
        self.assertEqual(error.error_code, "DATA_NOT_FOUND")
        self.assertEqual(error.details['entity_type'], 'Ticker')
        self.assertEqual(error.details['identifier'], 'INVALID')
    
    def test_data_integrity_error(self):
        """Test DataIntegrityError."""
        error = DataIntegrityError(
            constraint="unique_ticker_date",
            data={'ticker': 'AAPL', 'date': '2023-01-01'}
        )
        
        self.assertIn("Data integrity violation: unique_ticker_date", error.message)
        self.assertEqual(error.error_code, "DATA_INTEGRITY")
        self.assertEqual(error.details['constraint'], 'unique_ticker_date')


class TestBusinessRuleError(unittest.TestCase):
    """Test cases for BusinessRuleError."""
    
    def test_business_rule_error_creation(self):
        """Test business rule error creation."""
        context = {
            'account_id': 'ACC123',
            'action': 'BUY',
            'ticker': 'AAPL'
        }
        
        error = BusinessRuleError(
            rule="Maximum position size exceeded",
            context=context
        )
        
        self.assertIn("Business rule violation: Maximum position size exceeded", error.message)
        self.assertEqual(error.error_code, "BUSINESS_RULE")
        self.assertEqual(error.details['rule'], 'Maximum position size exceeded')
        self.assertEqual(error.details['context'], context)


class TestExternalServiceError(unittest.TestCase):
    """Test cases for ExternalServiceError."""
    
    def test_external_service_error(self):
        """Test external service error creation."""
        error = ExternalServiceError(
            service_name="IQFeed",
            operation="fetch_quotes",
            status_code=500
        )
        
        self.assertIn("External service error: IQFeed failed during fetch_quotes", error.message)
        self.assertEqual(error.error_code, "EXTERNAL_SERVICE")
        self.assertEqual(error.details['service_name'], 'IQFeed')
        self.assertEqual(error.details['operation'], 'fetch_quotes')
        self.assertEqual(error.details['status_code'], 500)


class TestRetryableError(unittest.TestCase):
    """Test cases for RetryableError."""
    
    def test_retryable_error_creation(self):
        """Test retryable error creation."""
        error = RetryableError(
            message="Temporary failure",
            max_retries=5
        )
        
        self.assertEqual(error.max_retries, 5)
        self.assertEqual(error.details['max_retries'], 5)
        self.assertIsInstance(error, TradingPlatformError)


class TestRateLimitError(unittest.TestCase):
    """Test cases for RateLimitError."""
    
    def test_rate_limit_error_creation(self):
        """Test rate limit error creation."""
        reset_time = datetime.now()
        
        error = RateLimitError(
            service="AlphaVantage",
            limit=5,
            reset_time=reset_time
        )
        
        self.assertIn("Rate limit exceeded for AlphaVantage: 5 requests", error.message)
        self.assertEqual(error.error_code, "RATE_LIMIT")
        self.assertEqual(error.details['service'], 'AlphaVantage')
        self.assertEqual(error.details['limit'], 5)
        self.assertEqual(error.details['reset_time'], reset_time.isoformat())
        self.assertIsInstance(error, RetryableError)


class TestSecurityErrors(unittest.TestCase):
    """Test cases for security-related errors."""
    
    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError(
            method="API_KEY",
            reason="invalid key format"
        )
        
        self.assertIn("Authentication failed using API_KEY: invalid key format", error.message)
        self.assertEqual(error.error_code, "AUTH_FAILED")
        self.assertEqual(error.details['method'], 'API_KEY')
        self.assertEqual(error.details['reason'], 'invalid key format')
        self.assertEqual(error.log_level, 'CRITICAL')
    
    def test_authorization_error(self):
        """Test AuthorizationError."""
        error = AuthorizationError(
            resource="trading_account",
            action="place_order",
            user="user123"
        )
        
        self.assertIn("Unauthorized access to trading_account for action: place_order", error.message)
        self.assertEqual(error.error_code, "AUTH_DENIED")
        self.assertEqual(error.details['resource'], 'trading_account')
        self.assertEqual(error.details['action'], 'place_order')
        self.assertEqual(error.details['user'], 'user123')


class TestMarketDataErrors(unittest.TestCase):
    """Test cases for market data domain errors."""
    
    def test_ticker_not_found_error(self):
        """Test TickerNotFoundError."""
        error = TickerNotFoundError(
            ticker="INVALID",
            exchange="NYSE"
        )
        
        self.assertIn("Ticker 'INVALID' not found on exchange 'NYSE'", error.message)
        self.assertEqual(error.error_code, "TICKER_NOT_FOUND")
        self.assertEqual(error.details['ticker'], 'INVALID')
        self.assertEqual(error.details['exchange'], 'NYSE')
    
    def test_market_data_unavailable_error(self):
        """Test MarketDataUnavailableError."""
        error = MarketDataUnavailableError(
            data_type="real_time_quotes",
            reason="market closed"
        )
        
        self.assertIn("Market data unavailable for real_time_quotes: market closed", error.message)
        self.assertEqual(error.error_code, "MARKET_DATA_UNAVAILABLE")
        self.assertEqual(error.details['data_type'], 'real_time_quotes')
        self.assertEqual(error.details['reason'], 'market closed')
    
    def test_stale_data_error(self):
        """Test StaleDataError."""
        last_update = datetime(2023, 1, 1, 10, 0, 0)
        
        with unittest.mock.patch('src.shared.exceptions.domain.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 11, 0, 0)
            
            error = StaleDataError(
                data_type="price_data",
                last_update=last_update,
                threshold_minutes=30
            )
        
        self.assertIn("Data for price_data is stale", error.message)
        self.assertEqual(error.error_code, "STALE_DATA")
        self.assertEqual(error.details['data_type'], 'price_data')
        self.assertEqual(error.details['threshold_minutes'], 30)


class TestTradingErrors(unittest.TestCase):
    """Test cases for trading domain errors."""
    
    def test_insufficient_funds_error(self):
        """Test InsufficientFundsError."""
        error = InsufficientFundsError(
            required=Decimal('10000.00'),
            available=Decimal('5000.00'),
            account_id="ACC123"
        )
        
        self.assertIn("Insufficient funds: required $10000.00, available $5000.00", error.message)
        self.assertEqual(error.error_code, "INSUFFICIENT_FUNDS")
        self.assertEqual(error.details['context']['required'], 10000.0)
        self.assertEqual(error.details['context']['available'], 5000.0)
        self.assertEqual(error.details['context']['shortfall'], 5000.0)
    
    def test_position_limit_exceeded_error(self):
        """Test PositionLimitExceededError."""
        error = PositionLimitExceededError(
            ticker="AAPL",
            current_position=800,
            requested=300,
            limit=1000
        )
        
        self.assertIn("Position limit exceeded for AAPL", error.message)
        self.assertEqual(error.error_code, "POSITION_LIMIT_EXCEEDED")
        self.assertEqual(error.details['context']['current_position'], 800)
        self.assertEqual(error.details['context']['requested'], 300)
        self.assertEqual(error.details['context']['total_would_be'], 1100)
        self.assertEqual(error.details['context']['limit'], 1000)
    
    def test_invalid_order_error(self):
        """Test InvalidOrderError."""
        order_details = {
            'ticker': 'AAPL',
            'quantity': -100,
            'price': 0
        }
        
        error = InvalidOrderError(
            order_type="MARKET",
            reason="negative quantity and zero price",
            order_details=order_details
        )
        
        self.assertIn("Invalid MARKET order: negative quantity and zero price", error.message)
        self.assertEqual(error.error_code, "INVALID_ORDER")
        self.assertEqual(error.details['context']['order_details'], order_details)
    
    def test_market_closed_error(self):
        """Test MarketClosedError."""
        current_time = datetime(2023, 1, 1, 20, 0, 0)  # 8 PM
        next_open = datetime(2023, 1, 2, 9, 30, 0)    # Next day 9:30 AM
        
        error = MarketClosedError(
            market="NYSE",
            current_time=current_time,
            next_open=next_open
        )
        
        self.assertIn("Market 'NYSE' is closed", error.message)
        self.assertIn("opens at", error.message)
        self.assertEqual(error.error_code, "MARKET_CLOSED")
        self.assertEqual(error.details['context']['market'], 'NYSE')


class TestTechnicalAnalysisErrors(unittest.TestCase):
    """Test cases for technical analysis domain errors."""
    
    def test_insufficient_data_error(self):
        """Test InsufficientDataError."""
        error = InsufficientDataError(
            indicator="SMA_50",
            required_periods=50,
            available_periods=30
        )
        
        self.assertIn("Insufficient data for SMA_50: requires 50 periods, have 30", error.message)
        self.assertEqual(error.error_code, "INSUFFICIENT_DATA")
        self.assertEqual(error.details['indicator'], 'SMA_50')
        self.assertEqual(error.details['required_periods'], 50)
        self.assertEqual(error.details['available_periods'], 30)
        self.assertEqual(error.details['shortfall'], 20)
    
    def test_invalid_indicator_parameters_error(self):
        """Test InvalidIndicatorParametersError."""
        parameters = {'period': -10, 'multiplier': 0}
        
        error = InvalidIndicatorParametersError(
            indicator="RSI",
            parameters=parameters,
            reason="period must be positive and multiplier cannot be zero"
        )
        
        self.assertIn("Invalid parameters for RSI", error.message)
        self.assertEqual(error.error_code, "INVALID_INDICATOR_PARAMS")
        self.assertEqual(error.details['parameters'], parameters)


class TestBacktestingErrors(unittest.TestCase):
    """Test cases for backtesting domain errors."""
    
    def test_backtest_data_error(self):
        """Test BacktestDataError."""
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 1, 1)
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        
        error = BacktestDataError(
            issue="missing price data for 30% of dates",
            start_date=start_date,
            end_date=end_date,
            tickers=tickers
        )
        
        self.assertIn("Backtest data issue: missing price data for 30% of dates", error.message)
        self.assertEqual(error.error_code, "BACKTEST_DATA_ERROR")
        self.assertEqual(error.details['tickers'], tickers)
        self.assertEqual(error.details['date_range_days'], 365)
    
    def test_strategy_execution_error(self):
        """Test StrategyExecutionError."""
        timestamp = datetime(2023, 1, 15, 14, 30, 0)
        
        error = StrategyExecutionError(
            strategy_name="MomentumStrategy",
            timestamp=timestamp,
            reason="division by zero in risk calculation"
        )
        
        self.assertIn("Strategy 'MomentumStrategy' failed at", error.message)
        self.assertEqual(error.error_code, "STRATEGY_EXECUTION_ERROR")
        self.assertEqual(error.details['strategy_name'], 'MomentumStrategy')
        self.assertEqual(error.details['timestamp'], timestamp.isoformat())


class TestDataProviderErrors(unittest.TestCase):
    """Test cases for data provider domain errors."""
    
    def test_iqfeed_error(self):
        """Test IQFeedError."""
        error = IQFeedError(
            operation="fetch_historical_data",
            error_message="Connection timeout"
        )
        
        self.assertIn("IQFeed error during fetch_historical_data: Connection timeout", error.message)
        self.assertEqual(error.error_code, "IQFEED_ERROR")
        self.assertEqual(error.details['provider'], 'IQFeed')
        self.assertEqual(error.details['operation'], 'fetch_historical_data')
    
    def test_alphavantage_error(self):
        """Test AlphaVantageError."""
        error = AlphaVantageError(
            endpoint="/query",
            status_code=429,
            response='{"Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day."}'
        )
        
        self.assertIn("Alpha Vantage API error: 429 from /query", error.message)
        self.assertEqual(error.error_code, "ALPHAVANTAGE_ERROR")
        self.assertEqual(error.details['provider'], 'AlphaVantage')
        self.assertEqual(error.details['status_code'], 429)


if __name__ == '__main__':
    unittest.main()