"""
Decorators for dependency injection configuration.

Provides convenient decorators for service registration,
lifecycle management, and dependency injection setup.
"""

from typing import Type, TypeVar, Callable, Any, Optional
from functools import wraps

from .container import ServiceLifetime, get_container

T = TypeVar('T')


def injectable(cls: Type[T]) -> Type[T]:
    """
    Mark a class as injectable and auto-register it.
    
    Usage:
        @injectable
        class MarketDataService:
            def __init__(self, config: MarketDataConfig):
                self.config = config
    """
    container = get_container()
    container.register(cls, cls)
    return cls


def singleton(service_type: Optional[Type] = None):
    """
    Register a service as singleton.
    
    Usage:
        @singleton()
        class DatabaseService:
            pass
            
        # Or with interface
        @singleton(IMarketDataProvider)
        class AlphaVantageProvider:
            pass
    """
    def decorator(cls: Type[T]) -> Type[T]:
        container = get_container()
        target_type = service_type if service_type else cls
        container.register(target_type, cls, ServiceLifetime.SINGLETON)
        return cls
    return decorator


def transient(service_type: Optional[Type] = None):
    """
    Register a service as transient (new instance each time).
    
    Usage:
        @transient()
        class OrderProcessor:
            pass
    """
    def decorator(cls: Type[T]) -> Type[T]:
        container = get_container()
        target_type = service_type if service_type else cls
        container.register(target_type, cls, ServiceLifetime.TRANSIENT)
        return cls
    return decorator


def scoped(service_type: Optional[Type] = None):
    """
    Register a service as scoped (one instance per scope).
    
    Usage:
        @scoped()
        class RequestProcessor:
            pass
    """
    def decorator(cls: Type[T]) -> Type[T]:
        container = get_container()
        target_type = service_type if service_type else cls
        container.register(target_type, cls, ServiceLifetime.SCOPED)
        return cls
    return decorator


def factory(service_type: Type[T], lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT):
    """
    Register a factory function for service creation.
    
    Usage:
        @factory(IMarketDataProvider)
        def create_market_data_provider():
            return AlphaVantageProvider(api_key="...")
    """
    def decorator(func: Callable[[], T]) -> Callable[[], T]:
        container = get_container()
        container.register_factory(service_type, func, lifetime)
        return func
    return decorator


def inject(*dependencies: Type):
    """
    Decorator for method-level dependency injection.
    
    Usage:
        class TradingService:
            @inject(IMarketDataProvider, ITradingConfig)
            def process_trade(self, symbol: str, market_data: IMarketDataProvider, config: ITradingConfig):
                pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            container = get_container()
            
            # Resolve dependencies
            injected_args = []
            for dep_type in dependencies:
                injected_args.append(container.resolve(dep_type))
            
            # Call original function with injected dependencies
            return func(*args, *injected_args, **kwargs)
        
        return wrapper
    return decorator


def lazy_inject(dependency_type: Type[T]) -> Callable[[], T]:
    """
    Create a lazy dependency resolver.
    
    Usage:
        class TradingService:
            def __init__(self):
                self._market_data = lazy_inject(IMarketDataProvider)
            
            def get_price(self, symbol: str):
                # Resolve dependency when first accessed
                market_data = self._market_data()
                return market_data.get_price(symbol)
    """
    def resolver() -> T:
        container = get_container()
        return container.resolve(dependency_type)
    
    return resolver


def configure_services(func: Callable[[Any], None]) -> Callable:
    """
    Decorator for service configuration functions.
    
    Usage:
        @configure_services
        def setup_services(container):
            container.register(IMarketDataProvider, AlphaVantageProvider, singleton())
            container.register(ITradingConfig, TradingConfig, singleton())
    """
    @wraps(func)
    def wrapper():
        container = get_container()
        return func(container)
    
    return wrapper


class Autowired:
    """
    Property descriptor for automatic dependency injection.
    
    Usage:
        class TradingService:
            market_data = Autowired(IMarketDataProvider)
            config = Autowired(ITradingConfig)
            
            def process_trade(self, symbol: str):
                price = self.market_data.get_price(symbol)
    """
    
    def __init__(self, dependency_type: Type[T]):
        self.dependency_type = dependency_type
        self._instance = None
        self._attr_name = None
    
    def __set_name__(self, owner: Type, name: str):
        self._attr_name = name
    
    def __get__(self, instance: Any, owner: Type) -> T:
        if instance is None:
            return self
        
        # Get or create instance
        attr_name = f"_{self._attr_name}_instance"
        if not hasattr(instance, attr_name):
            container = get_container()
            dependency_instance = container.resolve(self.dependency_type)
            setattr(instance, attr_name, dependency_instance)
        
        return getattr(instance, attr_name)
    
    def __set__(self, instance: Any, value: T):
        attr_name = f"_{self._attr_name}_instance"
        setattr(instance, attr_name, value)