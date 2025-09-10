# Logging Best Practices for Trading Platform

This guide provides comprehensive best practices for using the logging and error handling system in the trading platform.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Logging Levels and When to Use Them](#logging-levels)
3. [Structured Logging](#structured-logging)
4. [Context and Correlation IDs](#context-and-correlation-ids)
5. [Performance Logging](#performance-logging)
6. [Error Handling and Exception Logging](#error-handling)
7. [Domain-Specific Logging](#domain-specific-logging)
8. [Security and Audit Logging](#security-and-audit-logging)
9. [Performance Considerations](#performance-considerations)
10. [Common Anti-Patterns](#common-anti-patterns)
11. [Examples](#examples)

## Quick Start

### Replace Print Statements

**❌ Don't do this:**
```python
print(f"Processing ticker {ticker}")
print(f"Found {len(data)} records")
if error:
    print(f"ERROR: {error}")
```

**✅ Do this:**
```python
from src.infrastructure.logging import get_data_logger

logger = get_data_logger()
logger.info("Processing ticker", extra={'ticker': ticker})
logger.info("Data fetch completed", extra={'record_count': len(data)})
if error:
    logger.error("Data processing failed", exception=error)
```

### Basic Logger Setup

```python
# At the top of your script
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)  # Use __name__ for module-specific logger

# In your functions
def process_market_data(ticker):
    logger.info("Starting market data processing", extra={'ticker': ticker})
    # Your code here
    logger.info("Market data processing completed", extra={'ticker': ticker, 'duration': duration})
```

## Logging Levels

### DEBUG
- **Use for**: Detailed diagnostic information
- **Examples**: Variable values, function entry/exit, detailed flow
- **Audience**: Developers debugging issues

```python
logger.debug("Entering function", extra={
    'function': 'calculate_sma',
    'params': {'period': 20, 'data_points': len(prices)}
})
```

### INFO
- **Use for**: General information about program execution
- **Examples**: Process starts/completion, successful operations, business events
- **Audience**: Operations team, business users

```python
logger.info("Daily data processing started")
logger.info("Trade executed", extra={
    'ticker': 'AAPL', 
    'action': 'BUY', 
    'quantity': 100,
    'price': 150.25
})
```

### WARNING
- **Use for**: Something unexpected happened but system can continue
- **Examples**: Deprecated features, recoverable errors, configuration issues
- **Audience**: Operations team

```python
logger.warning("Using deprecated API endpoint", extra={
    'endpoint': '/api/v1/quotes',
    'replacement': '/api/v2/quotes',
    'deprecation_date': '2024-12-31'
})
```

### ERROR
- **Use for**: Error conditions that require attention but don't stop the program
- **Examples**: Failed operations, data validation errors, external service failures
- **Audience**: Operations team, developers

```python
logger.error("Failed to fetch ticker data", extra={
    'ticker': 'AAPL',
    'provider': 'IQFeed',
    'retry_count': 3
}, exception=e)
```

### CRITICAL
- **Use for**: Serious errors that may cause program termination
- **Examples**: Database connection failures, configuration errors, security breaches
- **Audience**: Operations team, management

```python
logger.critical("Database connection failed", extra={
    'database': 'markets_technicals',
    'host': 'localhost',
    'connection_attempts': 5
}, exception=e)
```

## Structured Logging

Always use the `extra` parameter to add structured data to your logs.

### Good Structure Examples

```python
# Trade logging
logger.info("Trade completed", extra={
    'trade_id': 'TRD-123456',
    'ticker': 'AAPL',
    'action': 'BUY',
    'quantity': 100,
    'price': 150.25,
    'total_value': 15025.0,
    'strategy': 'momentum',
    'account_id': 'ACC-789'
})

# Data processing
logger.info("Indicator calculation completed", extra={
    'indicator': 'RSI',
    'ticker': 'AAPL',
    'period': 14,
    'data_points': 252,
    'duration_ms': 45.2,
    'result_count': 238
})

# Error with context
logger.error("Data validation failed", extra={
    'validator': 'PriceRangeValidator',
    'ticker': 'AAPL',
    'date': '2023-01-15',
    'price': -10.50,
    'constraint': 'price_must_be_positive'
}, exception=e)
```

### Avoid These Structures

```python
# ❌ Don't embed data in message strings
logger.info(f"Processed ticker AAPL with 100 records in 1.5 seconds")

# ❌ Don't use non-serializable objects
logger.info("Processing complete", extra={'data': some_complex_object})

# ❌ Don't log sensitive information
logger.info("User login", extra={'password': user_password})  # NEVER DO THIS
```

## Context and Correlation IDs

Use context managers to track operations across multiple functions and scripts.

### Request/Operation Context

```python
from src.infrastructure.logging import set_request_context, get_logger

logger = get_logger(__name__)

def daily_batch_process():
    # Set context for the entire batch operation
    with set_request_context(
        request_id="BATCH_2023-01-15",
        user_id="scheduler",
        session_id="daily_batch"
    ):
        logger.info("Starting daily batch process")
        
        fetch_market_data()      # All logs will include context
        calculate_indicators()   # All logs will include context
        generate_signals()       # All logs will include context
        
        logger.info("Daily batch process completed")

def fetch_market_data():
    # This function's logs will automatically include the context
    logger.info("Fetching market data from IQFeed")
    # ... implementation
```

### Manual Context Setting

```python
from src.infrastructure.logging import LogContext

def process_ticker(ticker):
    # Set ticker-specific context
    with LogContext(ticker=ticker):
        logger.info("Processing ticker data")
        # All nested function calls will include ticker in context
```

## Performance Logging

### Automatic Performance Tracking

```python
from src.infrastructure.logging import log_performance

@log_performance(include_memory=True, include_cpu=True)
def calculate_technical_indicators(ticker_data):
    # Function automatically timed and logged
    # Memory and CPU usage tracked
    return indicators

# Alternative: manual timing
from src.infrastructure.logging import PerformanceTimer

def process_large_dataset():
    with PerformanceTimer('process_large_dataset', include_memory=True):
        # Your processing code here
        pass
```

### Batch Operation Logging

```python
from src.infrastructure.logging import BatchOperationTimer

def process_all_tickers(tickers):
    with BatchOperationTimer('process_all_tickers', len(tickers)) as timer:
        for i, ticker in enumerate(tickers):
            process_ticker(ticker)
            timer.update(1)  # Increment processed count
            
            if i % 100 == 0:
                logger.info(f"Processed {i+1}/{len(tickers)} tickers")
```

### Operation Tracking

```python
from src.infrastructure.logging import global_tracker

def fetch_ticker_data(ticker):
    with global_tracker.track('fetch_ticker_data'):
        # Your implementation
        return data

# Later, get statistics
stats = global_tracker.get_statistics('fetch_ticker_data')
logger.info("Operation statistics", extra={'stats': stats})
```

## Error Handling

### Use Domain-Specific Exceptions

```python
from src.shared.exceptions import TickerNotFoundError, InsufficientDataError
from src.infrastructure.logging import get_technical_logger

logger = get_technical_logger()

def calculate_sma(ticker, period, data):
    try:
        if not data:
            raise InsufficientDataError(
                indicator="SMA",
                required_periods=period,
                available_periods=0
            )
        
        if len(data) < period:
            raise InsufficientDataError(
                indicator="SMA",
                required_periods=period,
                available_periods=len(data)
            )
        
        # Calculate SMA
        return sum(data[-period:]) / period
        
    except InsufficientDataError as e:
        logger.error("Cannot calculate SMA", exception=e)
        raise  # Re-raise for caller to handle
    
    except Exception as e:
        logger.error("Unexpected error in SMA calculation", extra={
            'ticker': ticker,
            'period': period,
            'data_length': len(data) if data else 0
        }, exception=e)
        raise
```

### Exception Context Preservation

```python
def fetch_and_process_data(ticker):
    try:
        raw_data = fetch_raw_data(ticker)
        processed_data = process_data(raw_data)
        return processed_data
        
    except ExternalServiceError as e:
        # Add context to the error
        e.details['operation'] = 'fetch_and_process_data'
        e.details['ticker'] = ticker
        logger.error("Data fetch and processing failed", exception=e)
        raise
        
    except Exception as e:
        # Wrap unexpected errors
        wrapped_error = TradingPlatformError(
            message=f"Unexpected error processing {ticker}",
            error_code="PROCESSING_ERROR",
            cause=e,
            details={'ticker': ticker}
        )
        logger.error("Unexpected processing error", exception=wrapped_error)
        raise wrapped_error
```

## Domain-Specific Logging

### Use Appropriate Loggers

```python
# Data collection scripts
from src.infrastructure.logging import get_data_logger
logger = get_data_logger()

# Technical analysis scripts
from src.infrastructure.logging import get_technical_logger
logger = get_technical_logger()

# Trading execution scripts
from src.infrastructure.logging import get_trading_logger
logger = get_trading_logger()

# Market internals scripts
from src.infrastructure.logging import get_internals_logger
logger = get_internals_logger()

# Backtesting scripts
from src.infrastructure.logging import get_backtesting_logger
logger = get_backtesting_logger()
```

### Specialized Logging Methods

```python
# Trade logging (automatically goes to trade.log)
logger.log_trade(
    action="BUY",
    ticker="AAPL", 
    quantity=100,
    price=150.25,
    strategy="momentum",
    signal_strength=0.85
)

# Performance logging (automatically goes to performance.log)
logger.log_performance(
    operation="calculate_rsi",
    duration=0.045,
    ticker="AAPL",
    data_points=252
)

# Data fetch logging
logger.log_data_fetch(
    source="IQFeed",
    ticker="AAPL",
    records=1000,
    duration=2.5,
    errors=0
)
```

## Security and Audit Logging

### What to Audit Log

```python
# Configuration changes
logger.info("Configuration updated", extra={
    'config_key': 'database.connection_pool.size',
    'old_value': 10,
    'new_value': 20,
    'changed_by': 'admin_user',
    'change_reason': 'performance_optimization'
})

# Authentication events
logger.info("User authentication", extra={
    'user_id': 'trading_user',
    'auth_method': 'api_key',
    'source_ip': '192.168.1.100',
    'success': True
})

# Authorization events
logger.warning("Unauthorized access attempt", extra={
    'user_id': 'limited_user',
    'requested_resource': 'trading_account',
    'requested_action': 'place_order',
    'denied_reason': 'insufficient_permissions'
})

# All trades (automatically audited)
logger.log_trade("BUY", "AAPL", 100, 150.25)  # Goes to audit.log
```

### What NOT to Log

```python
# ❌ NEVER log passwords, API keys, or other secrets
logger.error("Login failed", extra={
    'username': username,
    'password': password  # NEVER DO THIS
})

# ❌ NEVER log personal information
logger.info("User data", extra={
    'ssn': user.ssn,      # NEVER DO THIS
    'phone': user.phone   # NEVER DO THIS
})

# ✅ DO log non-sensitive identifiers
logger.info("User operation", extra={
    'user_id': user.id,           # OK - internal ID
    'operation': 'view_portfolio', # OK - business operation
    'timestamp': datetime.now()    # OK - audit trail
})
```

## Performance Considerations

### Use Appropriate Log Levels

```python
# ❌ Expensive debug logging in production
logger.debug(f"Processing data: {expensive_calculation()}")  # Bad

# ✅ Use level checks for expensive operations
if logger.logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Processing data: {expensive_calculation()}")  # Good
```

### Batch Logging for High-Frequency Operations

```python
# ❌ Log every iteration in tight loops
for ticker in tickers:
    logger.info(f"Processing {ticker}")  # Too much logging

# ✅ Log periodically
for i, ticker in enumerate(tickers):
    if i % 100 == 0:
        logger.info("Batch progress", extra={
            'processed': i,
            'total': len(tickers),
            'current_ticker': ticker
        })
```

### Asynchronous Logging for High-Throughput

```python
# For very high-throughput scenarios, use buffered handler
from src.infrastructure.logging.handlers import BufferedAsyncHandler

# This buffers logs and writes them asynchronously
# Configure in logging setup
```

## Common Anti-Patterns

### ❌ String Formatting in Log Messages

```python
# Don't do this
logger.info(f"Processed ticker {ticker} with {count} records")

# Do this instead
logger.info("Ticker processing completed", extra={
    'ticker': ticker,
    'record_count': count
})
```

### ❌ Logging Inside Exception Handlers Without Re-raising

```python
# Don't swallow exceptions
try:
    risky_operation()
except Exception as e:
    logger.error("Operation failed", exception=e)
    return None  # Silent failure

# Do handle appropriately
try:
    risky_operation()
except RetryableError as e:
    logger.warning("Retryable error occurred", exception=e)
    # Implement retry logic
except Exception as e:
    logger.error("Operation failed", exception=e)
    raise  # Re-raise for caller to handle
```

### ❌ Generic Error Messages

```python
# Not helpful
logger.error("Something went wrong")

# Much better
logger.error("Technical indicator calculation failed", extra={
    'indicator': 'RSI',
    'ticker': 'AAPL',
    'period': 14,
    'data_points': len(data),
    'error_location': 'division_by_zero_check'
}, exception=e)
```

### ❌ Inconsistent Context

```python
# Inconsistent
logger.info("Starting process", extra={'symbol': 'AAPL'})
logger.info("Process complete", extra={'ticker': 'AAPL'})  # Different key name

# Consistent
logger.info("Starting process", extra={'ticker': 'AAPL'})
logger.info("Process complete", extra={'ticker': 'AAPL'})  # Same key name
```

## Examples

### Complete Script Example

```python
#!/usr/bin/env python3
"""
Example: Daily indicator calculation script with proper logging.
"""

from src.infrastructure.logging import get_technical_logger, set_request_context, log_performance
from src.infrastructure.database import get_technical_session
from src.shared.exceptions import InsufficientDataError, TickerNotFoundError
from datetime import datetime
import sys

logger = get_technical_logger()

@log_performance(include_memory=True)
def calculate_rsi(ticker, prices, period=14):
    """Calculate RSI with proper logging."""
    logger.debug("Starting RSI calculation", extra={
        'ticker': ticker,
        'period': period,
        'data_points': len(prices)
    })
    
    if len(prices) < period + 1:
        raise InsufficientDataError(
            indicator="RSI",
            required_periods=period + 1,
            available_periods=len(prices)
        )
    
    # RSI calculation logic here...
    rsi_value = 65.4  # Placeholder
    
    logger.debug("RSI calculation completed", extra={
        'ticker': ticker,
        'rsi_value': rsi_value,
        'data_points_used': period + 1
    })
    
    return rsi_value

def process_ticker(ticker):
    """Process a single ticker."""
    logger.info("Processing ticker", extra={'ticker': ticker})
    
    try:
        # Fetch data
        with get_technical_session() as session:
            prices = session.fetch_all(
                "SELECT close FROM prices WHERE ticker = %s ORDER BY date",
                (ticker,)
            )
        
        if not prices:
            raise TickerNotFoundError(ticker)
        
        logger.info("Data fetched successfully", extra={
            'ticker': ticker,
            'record_count': len(prices)
        })
        
        # Calculate indicators
        rsi = calculate_rsi(ticker, [p[0] for p in prices])
        
        # Store results
        with get_technical_session() as session:
            session.execute(
                "INSERT INTO indicators (ticker, date, rsi) VALUES (%s, %s, %s)",
                (ticker, datetime.now().date(), rsi)
            )
        
        logger.info("Ticker processing completed", extra={
            'ticker': ticker,
            'rsi': rsi
        })
        
        return True
        
    except TickerNotFoundError as e:
        logger.warning("Ticker not found", extra={'ticker': ticker}, exception=e)
        return False
        
    except InsufficientDataError as e:
        logger.warning("Insufficient data for calculation", extra={
            'ticker': ticker
        }, exception=e)
        return False
        
    except Exception as e:
        logger.error("Unexpected error processing ticker", extra={
            'ticker': ticker
        }, exception=e)
        raise

def main():
    """Main processing function."""
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    # Set context for the entire operation
    with set_request_context(
        request_id=f"RSI_CALC_{datetime.now().strftime('%Y%m%d_%H%M')}",
        user_id="scheduler"
    ):
        logger.info("Starting RSI calculation batch", extra={
            'ticker_count': len(tickers),
            'batch_id': f"RSI_CALC_{datetime.now().strftime('%Y%m%d_%H%M')}"
        })
        
        success_count = 0
        
        for ticker in tickers:
            if process_ticker(ticker):
                success_count += 1
        
        logger.info("RSI calculation batch completed", extra={
            'total_tickers': len(tickers),
            'successful': success_count,
            'failed': len(tickers) - success_count,
            'success_rate': success_count / len(tickers)
        })
        
        # Log performance summary
        from src.infrastructure.logging import global_tracker
        global_tracker.log_statistics()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical("Script execution failed", exception=e)
        sys.exit(1)
```

### Migration Example

**Before (old logging):**
```python
import logging
import psycopg2
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_data():
    print("Starting data processing...")
    
    conn = psycopg2.connect(
        database="markets_technicals",
        user="postgres", 
        password="root",
        host="localhost"
    )
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM prices")
        data = cursor.fetchall()
        print(f"Fetched {len(data)} records")
        
        # Process data
        results = []
        for row in data:
            try:
                result = calculate_indicator(row)
                results.append(result)
            except Exception as e:
                print(f"Error processing row: {e}")
                
        print(f"Processed {len(results)} records successfully")
        
    except Exception as e:
        print(f"Database error: {e}")
    finally:
        conn.close()
```

**After (new logging):**
```python
from src.infrastructure.logging import get_technical_logger, set_request_context, log_performance
from src.infrastructure.database import get_technical_session
from src.shared.exceptions import DataProcessingError
from datetime import datetime

logger = get_technical_logger()

@log_performance(include_memory=True)
def process_data():
    """Process market data with proper logging and error handling."""
    
    with set_request_context(
        request_id=f"DATA_PROC_{datetime.now().strftime('%Y%m%d_%H%M')}"
    ):
        logger.info("Starting data processing")
        
        try:
            with get_technical_session() as session:
                data = session.fetch_all("SELECT * FROM prices")
                
                logger.info("Data fetch completed", extra={
                    'record_count': len(data)
                })
                
                # Process data
                results = []
                errors = []
                
                for i, row in enumerate(data):
                    try:
                        result = calculate_indicator(row)
                        results.append(result)
                        
                        if i % 1000 == 0:
                            logger.debug("Processing progress", extra={
                                'processed': i,
                                'total': len(data),
                                'success_count': len(results),
                                'error_count': len(errors)
                            })
                            
                    except Exception as e:
                        errors.append({'row_id': row[0], 'error': str(e)})
                        logger.warning("Row processing failed", extra={
                            'row_id': row[0] if row else None
                        }, exception=e)
                
                logger.info("Data processing completed", extra={
                    'total_records': len(data),
                    'successful': len(results),
                    'failed': len(errors),
                    'success_rate': len(results) / len(data) if data else 0
                })
                
                if errors:
                    logger.warning("Processing completed with errors", extra={
                        'error_count': len(errors),
                        'error_rate': len(errors) / len(data)
                    })
                
                return results
                
        except Exception as e:
            logger.error("Data processing failed", exception=e)
            raise DataProcessingError(
                message="Failed to process market data",
                cause=e,
                details={'operation': 'process_data'}
            )
```

This comprehensive logging system replaces all print statements and provides professional-grade observability for your trading platform!