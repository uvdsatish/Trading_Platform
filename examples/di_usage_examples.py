"""
Examples of how to use the dependency injection system in the trading platform.

Demonstrates service registration, dependency resolution, and integration
with configuration and database systems.
"""

from typing import List, Optional
from abc import ABC, abstractmethod

from src.infrastructure.di import (
    Container,
    ServiceLifetime,
    injectable,
    singleton,
    transient,
    scoped,
    Autowired
)
from src.infrastructure.di.integration import (
    create_configured_container,
    bootstrap_application,
    database_service,
    trading_service
)
from src.config import TradingConfig, IQFeedConfig
from src.infrastructure.database import IConnectionPool
from src.infrastructure.logging import TradingLogger


# Example 1: Basic Service Registration and Resolution
def basic_container_example():
    """Demonstrate basic container usage."""
    
    # Create container
    container = Container()
    
    # Define services
    class DatabaseService:
        def __init__(self):
            self.connected = True
        
        def query(self, sql: str) -> List[dict]:
            return [{"result": "data"}]
    
    class MarketDataService:
        def __init__(self, db: DatabaseService):
            self.db = db
        
        def get_price(self, symbol: str) -> float:
            data = self.db.query(f"SELECT price FROM prices WHERE symbol='{symbol}'")
            return 100.0
    
    # Register services
    container.register(DatabaseService, DatabaseService, ServiceLifetime.SINGLETON)
    container.register(MarketDataService, MarketDataService, ServiceLifetime.TRANSIENT)
    
    # Resolve services (automatic dependency injection)
    market_service = container.resolve(MarketDataService)
    price = market_service.get_price("AAPL")
    
    print(f"AAPL Price: ${price}")


# Example 2: Interface-based Registration
def interface_example():
    """Demonstrate interface-based service registration."""
    
    # Define interfaces
    class IMarketDataProvider(ABC):
        @abstractmethod
        def get_price(self, symbol: str) -> float:
            pass
    
    class IOrderService(ABC):
        @abstractmethod
        def place_order(self, symbol: str, quantity: int, price: float) -> str:
            pass
    
    # Implement concrete classes
    class AlphaVantageProvider(IMarketDataProvider):
        def __init__(self, config: TradingConfig):
            self.config = config
            self.api_key = config.data_providers.get('alpha_vantage', {}).get('api_key')
        
        def get_price(self, symbol: str) -> float:
            # Mock implementation
            return 150.0
    
    class TradingOrderService(IOrderService):
        def __init__(self, market_data: IMarketDataProvider, logger: TradingLogger):
            self.market_data = market_data
            self.logger = logger
        
        def place_order(self, symbol: str, quantity: int, price: float) -> str:
            current_price = self.market_data.get_price(symbol)
            order_id = f"ORDER_{symbol}_{quantity}"
            
            self.logger.log_trade(
                action="BUY" if quantity > 0 else "SELL",
                ticker=symbol,
                quantity=abs(quantity),
                price=price,
                order_id=order_id
            )
            
            return order_id
    
    # Create configured container
    container = create_configured_container()
    
    # Register business services
    container.register(IMarketDataProvider, AlphaVantageProvider, ServiceLifetime.SINGLETON)
    container.register(IOrderService, TradingOrderService, ServiceLifetime.TRANSIENT)
    
    # Resolve and use services
    order_service = container.resolve(IOrderService)
    order_id = order_service.place_order("TSLA", 100, 800.0)
    
    print(f"Order placed: {order_id}")


# Example 3: Decorator-based Registration
def decorator_example():
    """Demonstrate decorator-based service registration."""
    
    container = Container()
    
    # Auto-register with decorators
    @singleton()
    class ConfigurationManager:
        def __init__(self):
            self.settings = {"api_timeout": 30}
    
    @transient()
    class DataProcessor:
        def __init__(self, config: ConfigurationManager):
            self.config = config
            self.timeout = config.settings["api_timeout"]
        
        def process(self, data: dict) -> dict:
            return {"processed": data, "timeout": self.timeout}
    
    # Injectable service (auto-registers itself)
    @container.injectable
    class ReportGenerator:
        def __init__(self, processor: DataProcessor):
            self.processor = processor
        
        def generate_report(self, raw_data: dict) -> dict:
            processed = self.processor.process(raw_data)
            return {"report": processed}
    
    # Resolve services
    report_gen = container.resolve(ReportGenerator)
    report = report_gen.generate_report({"symbol": "AAPL", "price": 150})
    
    print(f"Generated report: {report}")


# Example 4: Scoped Services
def scoped_services_example():
    """Demonstrate scoped service lifetime."""
    
    container = Container()
    
    @scoped()
    class RequestContext:
        def __init__(self):
            self.request_id = f"req_{id(self)}"
            self.start_time = "2024-01-01T10:00:00"
    
    @transient()
    class OrderProcessor:
        def __init__(self, context: RequestContext):
            self.context = context
        
        def process_order(self, symbol: str) -> dict:
            return {
                "symbol": symbol,
                "request_id": self.context.request_id,
                "start_time": self.context.start_time
            }
    
    # Use scoped services
    with container.scope() as scope:
        # Both processors will share the same RequestContext instance
        processor1 = container.resolve(OrderProcessor, scope)
        processor2 = container.resolve(OrderProcessor, scope)
        
        order1 = processor1.process_order("AAPL")
        order2 = processor2.process_order("TSLA")
        
        print(f"Order 1 request ID: {order1['request_id']}")
        print(f"Order 2 request ID: {order2['request_id']}")
        print(f"Same request ID: {order1['request_id'] == order2['request_id']}")


# Example 5: Property Injection with Autowired
def autowired_example():
    """Demonstrate property-based dependency injection."""
    
    container = create_configured_container()
    
    class TradingStrategy:
        # Properties are automatically injected
        config = Autowired(TradingConfig)
        logger = Autowired(TradingLogger)
        db_pool = Autowired(IConnectionPool)
        
        def __init__(self, strategy_name: str):
            self.strategy_name = strategy_name
        
        def execute_trade(self, symbol: str, quantity: int):
            # Access injected dependencies
            self.logger.info(f"Executing {self.strategy_name} for {symbol}")
            
            # Use configuration
            max_position = self.config.risk_management.max_position_size
            if abs(quantity) > max_position:
                self.logger.warning(f"Position size {quantity} exceeds maximum {max_position}")
                return False
            
            # Use database connection
            with self.db_pool.get_connection() as conn:
                # Execute trade logic
                self.logger.log_trade(
                    action="BUY" if quantity > 0 else "SELL",
                    ticker=symbol,
                    quantity=abs(quantity),
                    price=100.0  # Mock price
                )
            
            return True
    
    # Create strategy instance (dependencies are injected automatically)
    strategy = TradingStrategy("MomentumStrategy")
    success = strategy.execute_trade("AAPL", 100)
    
    print(f"Trade executed successfully: {success}")


# Example 6: Database Service Integration
def database_integration_example():
    """Demonstrate database service integration."""
    
    container = bootstrap_application()
    
    @database_service('technical_analysis')
    class TechnicalAnalysisService:
        def __init__(self, db_pool: IConnectionPool, logger: TradingLogger):
            self.db_pool = db_pool
            self.logger = logger
        
        def calculate_sma(self, symbol: str, period: int) -> Optional[float]:
            """Calculate Simple Moving Average."""
            with self.db_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT AVG(close) as sma
                    FROM usstockseod 
                    WHERE ticker = %s 
                    ORDER BY date DESC 
                    LIMIT %s
                """, (symbol, period))
                
                result = cursor.fetchone()
                sma = result[0] if result and result[0] else None
                
                self.logger.debug(f"Calculated {period}-day SMA for {symbol}: {sma}")
                return sma
    
    @trading_service()
    class MomentumTradingService:
        def __init__(self, 
                     technical_service: TechnicalAnalysisService, 
                     config: TradingConfig,
                     logger: TradingLogger):
            self.technical_service = technical_service
            self.config = config
            self.logger = logger
        
        def generate_signals(self, symbol: str) -> List[dict]:
            """Generate trading signals based on momentum."""
            sma_20 = self.technical_service.calculate_sma(symbol, 20)
            sma_50 = self.technical_service.calculate_sma(symbol, 50)
            
            if sma_20 and sma_50:
                if sma_20 > sma_50:
                    signal = {
                        "symbol": symbol,
                        "action": "BUY",
                        "signal_strength": (sma_20 - sma_50) / sma_50,
                        "sma_20": sma_20,
                        "sma_50": sma_50
                    }
                    self.logger.info(f"Generated BUY signal for {symbol}")
                    return [signal]
            
            return []
    
    # Register and resolve services
    container.register(TechnicalAnalysisService, TechnicalAnalysisService, ServiceLifetime.SINGLETON)
    container.register(MomentumTradingService, MomentumTradingService, ServiceLifetime.TRANSIENT)
    
    # Use the services
    trading_service = container.resolve(MomentumTradingService)
    signals = trading_service.generate_signals("AAPL")
    
    print(f"Generated {len(signals)} trading signals")
    for signal in signals:
        print(f"  Signal: {signal}")


# Example 7: Factory Services
def factory_services_example():
    """Demonstrate factory-based service creation."""
    
    container = create_configured_container()
    
    class StrategyFactory:
        def __init__(self, config: TradingConfig, logger: TradingLogger):
            self.config = config
            self.logger = logger
        
        def create_momentum_strategy(self) -> 'MomentumStrategy':
            return MomentumStrategy(
                lookback_period=self.config.strategies.get('momentum_lookback', 20),
                logger=self.logger
            )
        
        def create_mean_reversion_strategy(self) -> 'MeanReversionStrategy':
            return MeanReversionStrategy(
                threshold=self.config.strategies.get('mean_reversion_threshold', 2.0),
                logger=self.logger
            )
    
    class MomentumStrategy:
        def __init__(self, lookback_period: int, logger: TradingLogger):
            self.lookback_period = lookback_period
            self.logger = logger
    
    class MeanReversionStrategy:
        def __init__(self, threshold: float, logger: TradingLogger):
            self.threshold = threshold
            self.logger = logger
    
    # Register factory
    container.register(StrategyFactory, StrategyFactory, ServiceLifetime.SINGLETON)
    
    # Register factory methods for specific strategies
    def create_momentum_strategy():
        factory = container.resolve(StrategyFactory)
        return factory.create_momentum_strategy()
    
    def create_mean_reversion_strategy():
        factory = container.resolve(StrategyFactory)
        return factory.create_mean_reversion_strategy()
    
    container.register_factory(MomentumStrategy, create_momentum_strategy, ServiceLifetime.TRANSIENT)
    container.register_factory(MeanReversionStrategy, create_mean_reversion_strategy, ServiceLifetime.TRANSIENT)
    
    # Resolve strategies
    momentum = container.resolve(MomentumStrategy)
    mean_reversion = container.resolve(MeanReversionStrategy)
    
    print(f"Created momentum strategy with lookback: {momentum.lookback_period}")
    print(f"Created mean reversion strategy with threshold: {mean_reversion.threshold}")


def main():
    """Run all examples."""
    print("=== Basic Container Example ===")
    basic_container_example()
    
    print("\n=== Interface Example ===")
    interface_example()
    
    print("\n=== Decorator Example ===")
    decorator_example()
    
    print("\n=== Scoped Services Example ===")
    scoped_services_example()
    
    print("\n=== Autowired Example ===")
    autowired_example()
    
    print("\n=== Database Integration Example ===")
    database_integration_example()
    
    print("\n=== Factory Services Example ===")
    factory_services_example()
    
    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    main()