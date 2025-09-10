# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Transformation Status

This trading platform is undergoing a comprehensive **Object-Oriented Architecture Transformation** from procedural "spaghetti code" with 40+ scripts to a modern Domain-Driven Design (DDD) architecture.

### ✅ Phase 1, Week 1 - Infrastructure Foundation (COMPLETED)
- **Configuration Management**: Centralized YAML/environment-based configuration system
- **Database Connection Pooling**: Thread-safe multi-database support with transaction management  
- **Logging & Error Handling**: Structured JSON logging with domain-specific exception hierarchy
- **Dependency Injection**: Full DI container with automatic dependency resolution

### ✅ Phase 1, Week 2 - Domain Models & Base Classes (95% COMPLETED)
- **Domain Entities**: Complete business entities (Ticker, OHLCV, Order, Position, Portfolio, Indicator, Signal, Strategy)
- **Value Objects**: Immutable domain values (Price, Money, Quantity, IndicatorValue, SignalCondition, etc.)
- **Specifications**: Business rule objects for querying and filtering domain entities
- **Domain Events**: Event-driven architecture for cross-domain communication

### 🚧 Current Phase: Phase 1, Week 2 - Final Components (IN PROGRESS)
- Repository patterns for data access (next)
- Domain services implementation
- Unit testing framework

## New OOP Architecture

### Modern Infrastructure (src/)
```
src/
├── config/                          # Centralized configuration management
│   ├── base.py                      # Base configuration classes
│   ├── database.py                  # Multi-database configuration
│   ├── data_providers.py            # IQFeed, Alpha Vantage configs
│   └── application.yaml             # Main configuration file
├── infrastructure/
│   ├── database/                    # Connection pooling & transaction management
│   │   ├── connection_pool.py       # Thread-safe PostgreSQL pools
│   │   ├── session_factory.py      # Database session management
│   │   └── transaction_manager.py  # Nested transaction support
│   ├── logging/                     # Structured logging system
│   │   ├── base.py                  # TradingLogger with JSON output
│   │   ├── formatters.py            # Multiple log formatters
│   │   ├── handlers.py              # Custom file/rotation handlers
│   │   └── performance.py           # Performance tracking decorators
│   └── di/                         # Dependency injection container
│       ├── container.py            # Main DI container
│       ├── decorators.py           # @singleton, @transient decorators
│       └── integration.py          # Bootstrap with core services
├── shared/
│   └── exceptions/                 # Domain-specific exception hierarchy
│       ├── base.py                 # TradingPlatformError base classes
│       └── domain.py               # MarketDataError, TradingError, etc.
└── domain/                        # Domain models (95% COMPLETE)
    ├── shared/                    # Base domain classes (Entity, ValueObject, Specification)
    ├── market_data/               # Market data entities (Ticker, OHLCV, Quote, MarketSession)
    ├── trading/                   # Trading domain (Order, Position, Portfolio, Trade, Account)
    ├── technical_analysis/        # Technical analysis (Indicator, Signal, Strategy, Pattern)
    └── backtesting/               # Backtesting domain models (future)
```

### Usage Examples

#### Configuration Service
```python
from src.config import ConfigurationService

config = ConfigurationService()
db_config = config.get_multi_database_config()
trading_config = config.get_trading_config()
```

#### Database Connection Pool
```python
from src.infrastructure.database import get_connection_pool

pool = get_connection_pool('technical_analysis')
with pool.get_connection() as conn:
    # Use connection with automatic cleanup
    pass
```

#### Structured Logging
```python
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)
logger.log_trade(action="BUY", ticker="AAPL", quantity=100, price=150.0)
```

#### Dependency Injection
```python
from src.infrastructure.di import Container, singleton
from src.config import ConfigurationService
from src.infrastructure.logging import TradingLogger

container = Container()

@singleton()
class MarketDataService:
    def __init__(self, config: ConfigurationService, logger: TradingLogger):
        self.config = config
        self.logger = logger

service = container.resolve(MarketDataService)  # Auto-injected dependencies
```

#### Domain Models Usage
```python
# Modern domain-driven approach replaces scattered scripts
from src.domain.market_data import Ticker, OHLCV, TickerId
from src.domain.trading import Order, Position, Portfolio, OrderType, OrderSide
from src.domain.technical_analysis import Strategy, Signal, Indicator, IndicatorType

# Create domain objects with built-in validation
ticker = Ticker(id=TickerId("AAPL"), symbol="AAPL", company_name="Apple Inc")
strategy = Strategy(id=StrategyId("runaway"), name="RunAway Momentum Strategy")

# Business logic in domain objects
if ticker.is_listed():
    signal = strategy.generate_signal(market_data)
    if signal.is_bullish() and signal.strength.is_strong():
        order = portfolio.place_order(ticker_id, OrderSide.BUY, quantity)
```

## Legacy Commands (Being Migrated)

### Data Collection Scripts
```bash
# Install dependencies
pip install -r requirements.txt

# Run main data collection scripts
python Data_Management/Combined_iqfeed_upload.py      # Historical data download
python Data_Management/IQDelta.py                     # Incremental data updates  
python Data_Management/tickers_update.py              # Update ticker lists
```

### Trading Strategy Scripts  
```bash
# Run core trading strategies
python Technicals/RunAway.py                          # Current momentum strategy
python Technicals/RunAway-historical.py              # Historical backtesting
python Utils/breakout_scanner.py                      # 52-week breakout scanner
python Utils/breakout_scanner_25day.py               # 25-day breakout scanner
python Utils/breakout_scanner_100day.py              # 100-day breakout scanner
```

### Technical Analysis Scripts
```bash
# Run technical analysis
python Technicals/Plurality-WAMRS/Plurality-RS-upload.py        # Relative strength analysis
python Technicals/Key_Indicators_population/KeyIndicatorsPopulation.py         # Technical indicators
python Technicals/Key_Indicators_population/KeyIndicatorsPopulation_Delta_Spark.py  # Spark-based indicators
python Technicals/Daily_Tech_Criteria.py             # Daily screening criteria
```

### Market Internals Scripts
```bash
# Run market internals analysis
python Internals/Hindenburg_Omen/HindenburgOmen_model.py       # Market crash indicator
python Internals/Hindenburg_Omen/data_loading.py              # Load internal market data
```

### Performance Analysis Scripts
```bash
# Performance and trade analysis
python Technicals/Performance_analysis/Combined_performance_SPY.py      # SPY performance analysis
python Technicals/TradeLog_Pre.py                     # Pre-trade analysis
python Technicals/TradeLog_Post.py                    # Post-trade analysis
```

## Architecture Overview

### Core Modules
- **Data_Management/**: Market data acquisition via IQFeed, PostgreSQL storage, ticker management
- **Technicals/**: Technical analysis, trading strategies (RunAway, Plurality-WAMRS), breakout detection
- **Internals/**: Market breadth analysis, Hindenburg Omen crash indicator
- **Utils/**: Scanners, utility functions, data processing tools
- **Back_Testing/**: Performance analysis and backtesting frameworks

### Key Systems
- **Plurality-WAMRS**: Proprietary relative strength system for sector rotation analysis
- **RunAway Strategy**: Momentum-based trading system with historical backtesting capabilities  
- **IQFeed Integration**: Real-time and historical market data via pyiqfeed library
- **PostgreSQL Storage**: Multiple databases (markets_technicals, markets_internals, Plurality)
- **Spark Support**: Large-scale technical indicator calculations via PySpark

### Data Flow
1. **Data Ingestion**: Combined_iqfeed_upload.py → PostgreSQL (usstockseod table)
2. **Delta Updates**: IQDelta.py for incremental data updates
3. **Technical Analysis**: Key_Indicators_population/ modules → key_indicators tables
4. **Strategy Execution**: RunAway.py reads indicators → generates trade signals
5. **Performance Tracking**: TradeLog_Pre/Post.py for trade analysis

## Modern Configuration System

### New Configuration (YAML-based)
```yaml
# src/config/application.yaml
environment: development
logging:
  level: INFO
  format: json
  
databases:
  technical_analysis:
    host: localhost
    port: 5432
    database: markets_technicals
    username: ${DB_USER}
    password: ${DB_PASSWORD}
  internals:
    host: localhost  
    port: 5432
    database: markets_internals
    username: ${DB_USER}
    password: ${DB_PASSWORD}
```

### Legacy Configuration (Being Phased Out)
- Copy `config_template.py` to `config.py` with database and IQFeed credentials  
- Set up `pyiqfeed/localconfig/passwords.py` and `Data_Management/pyiqfeed/localconfig/passwords.py`
- Configure PostgreSQL databases: markets_technicals, markets_internals, Plurality

## Testing & Quality Assurance

### Run Tests
```bash
# Run unit tests for new OOP components
python -m pytest tests/unit/ -v

# Run integration tests  
python -m pytest tests/integration/ -v

# Run specific test modules
python -m pytest tests/unit/infrastructure/test_logging.py -v
python -m pytest tests/unit/shared/test_exceptions.py -v
```

### Code Quality
```bash
# Type checking (if mypy is installed)
mypy src/

# Code formatting (if black is installed)
black src/ tests/

# Linting (if flake8 is installed) 
flake8 src/ tests/
```

## Database Schema Patterns

### Core Tables
- **usstockseod**: Core OHLCV data with date, ticker, open, high, low, close, volume
- **key_indicators_***: Technical indicators with date, ticker, and calculated metrics
- **rs_industry_groups**: Relative strength by industry group and date
- Market internal tables follow *_composite_raw naming pattern

### Multi-Database Architecture
- **markets_technicals**: Technical analysis data and indicators
- **markets_internals**: Market breadth and internal data
- **Plurality**: Proprietary relative strength calculations
- **fundamentals**: Fundamental analysis data (future)
- **macro**: Macroeconomic indicators (future)
- **backtesting**: Backtesting results and performance metrics (future)

## Dependencies

### Core Infrastructure
- pandas, psycopg2-binary, numpy, pyspark
- pyyaml (configuration), python-dotenv (environment variables)
- threading, contextlib, abc (built-in Python modules)

### Analysis & Visualization
- matplotlib, seaborn, scipy

### Data Processing  
- openpyxl, xlrd

### System & Monitoring
- psutil for monitoring

### Development & Testing
- pytest (unit testing), pytest-asyncio
- mypy (type checking), black (formatting), flake8 (linting)

## Migration Strategy

### Phase 1: Infrastructure Foundation ✅ COMPLETED
- [x] Configuration Management System
- [x] Database Connection Pooling  
- [x] Logging and Error Handling
- [x] Dependency Injection Container

### Phase 2: Domain Models ✅ 95% COMPLETED
- [x] Market Data domain models (Ticker, OHLCV, Quote, MarketSession + value objects & specifications)
- [x] Trading domain models (Order, Position, Portfolio, Trade, Account + value objects & specifications)
- [x] Technical Analysis domain models (Indicator, Signal, Strategy, Pattern + value objects & specifications)
- [ ] Repository patterns for data access (5% remaining)

### Phase 3: Service Layer (Future)
- [ ] Domain services
- [ ] Application services
- [ ] Integration services

### Phase 4: Legacy Migration (Future)
- [ ] Migrate existing 40+ scripts to use new architecture
- [ ] Refactor data collection scripts
- [ ] Refactor trading strategy scripts
- [ ] Refactor analysis scripts

## Development Best Practices

### New Code Guidelines
1. **Use Dependency Injection**: Register services with DI container
2. **Structured Logging**: Use TradingLogger with context and trade logging
3. **Configuration Management**: Access config via ConfigurationService
4. **Database Access**: Use connection pools and transaction management
5. **Exception Handling**: Use domain-specific exceptions from shared.exceptions
6. **Type Hints**: Provide complete type annotations for all new code
7. **Testing**: Write unit tests for all new components

### Legacy Code Guidelines
- All scripts expect PostgreSQL connection via config.py parameters
- IQFeed requires active subscription and running client  
- Spark scripts (KeyIndicatorsPopulation_Delta_Spark.py) need PySpark configuration
- Date formats are typically YYYYMMDD strings for IQFeed compatibility
- Chunked processing (default 500 tickers) for large datasets