"""
Dependency injection container for the trading platform.

Provides automatic dependency resolution, service lifecycle management,
and configuration-based service registration.

Usage Examples:
    # Basic service registration
    from src.infrastructure.di import Container, singleton, transient
    
    container = Container()
    container.register(IMarketDataProvider, AlphaVantageProvider, singleton())
    
    # Automatic dependency injection
    @container.injectable
    class TradingService:
        def __init__(self, market_data: IMarketDataProvider, config: TradingConfig):
            self.market_data = market_data
            self.config = config
    
    # Service resolution
    trading_service = container.resolve(TradingService)
    
    # Context-based resolution
    with container.scope() as scope:
        # Services with scoped lifetime
        scoped_service = scope.resolve(IScopedService)
"""

from .container import (
    Container,
    ServiceDescriptor,
    ServiceScope,
    ServiceLifetime,
    DependencyResolutionError,
    CircularDependencyError,
    ServiceNotRegisteredError
)

from .decorators import (
    injectable,
    singleton,
    transient,
    scoped,
    factory
)

from .providers import (
    ServiceProvider,
    SingletonProvider,
    TransientProvider,
    ScopedProvider,
    FactoryProvider
)

from .registry import (
    ServiceRegistry,
    ServiceRegistration
)

__all__ = [
    # Core container
    'Container',
    'ServiceDescriptor',
    'ServiceScope',
    'ServiceLifetime',
    
    # Exceptions
    'DependencyResolutionError',
    'CircularDependencyError',
    'ServiceNotRegisteredError',
    
    # Decorators
    'injectable',
    'singleton',
    'transient',
    'scoped',
    'factory',
    
    # Providers
    'ServiceProvider',
    'SingletonProvider',
    'TransientProvider',
    'ScopedProvider',
    'FactoryProvider',
    
    # Registry
    'ServiceRegistry',
    'ServiceRegistration',
]