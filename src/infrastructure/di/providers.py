"""
Service providers for different lifetime management strategies.

Implements the provider pattern for service instantiation
with support for singleton, transient, scoped, and factory lifetimes.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Type, TypeVar, Optional
import threading
from weakref import WeakKeyDictionary

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class ServiceProvider(ABC):
    """Abstract base class for service providers."""
    
    def __init__(self, service_type: Type[T], implementation: Type[T]):
        self.service_type = service_type
        self.implementation = implementation
    
    @abstractmethod
    def get_instance(self, container: 'Container', **kwargs) -> T:
        """Get service instance."""
        pass
    
    @abstractmethod
    def dispose(self):
        """Dispose provider resources."""
        pass


class SingletonProvider(ServiceProvider):
    """Provides singleton instances (one per container)."""
    
    def __init__(self, service_type: Type[T], implementation: Type[T]):
        super().__init__(service_type, implementation)
        self._instance: Optional[T] = None
        self._lock = threading.Lock()
        self._created = False
    
    def get_instance(self, container: 'Container', **kwargs) -> T:
        if not self._created:
            with self._lock:
                if not self._created:
                    self._instance = self._create_instance(container)
                    self._created = True
                    logger.debug(f"Created singleton: {self.service_type.__name__}")
        
        return self._instance
    
    def _create_instance(self, container: 'Container') -> T:
        """Create the singleton instance."""
        from .container import Container  # Avoid circular import
        
        if not isinstance(container, Container):
            raise ValueError("Container must be a Container instance")
        
        # Use container's dependency resolution
        return container._create_instance_from_type(self.implementation)
    
    def dispose(self):
        with self._lock:
            if self._instance and hasattr(self._instance, 'dispose'):
                try:
                    self._instance.dispose()
                except Exception as e:
                    logger.warning(f"Error disposing singleton {self.service_type.__name__}: {e}")
            
            self._instance = None
            self._created = False


class TransientProvider(ServiceProvider):
    """Provides transient instances (new instance every time)."""
    
    def get_instance(self, container: 'Container', **kwargs) -> T:
        instance = self._create_instance(container)
        logger.debug(f"Created transient: {self.service_type.__name__}")
        return instance
    
    def _create_instance(self, container: 'Container') -> T:
        """Create a new transient instance."""
        from .container import Container  # Avoid circular import
        
        if not isinstance(container, Container):
            raise ValueError("Container must be a Container instance")
        
        return container._create_instance_from_type(self.implementation)
    
    def dispose(self):
        pass  # Nothing to dispose for transient


class ScopedProvider(ServiceProvider):
    """Provides scoped instances (one per scope)."""
    
    def __init__(self, service_type: Type[T], implementation: Type[T]):
        super().__init__(service_type, implementation)
        self._scoped_instances: WeakKeyDictionary = WeakKeyDictionary()
        self._lock = threading.Lock()
    
    def get_instance(self, container: 'Container', scope=None, **kwargs) -> T:
        if scope is None:
            raise ValueError("Scoped services require a scope")
        
        with self._lock:
            if scope not in self._scoped_instances:
                self._scoped_instances[scope] = self._create_instance(container)
                logger.debug(f"Created scoped: {self.service_type.__name__}")
            
            return self._scoped_instances[scope]
    
    def _create_instance(self, container: 'Container') -> T:
        """Create a scoped instance."""
        from .container import Container  # Avoid circular import
        
        if not isinstance(container, Container):
            raise ValueError("Container must be a Container instance")
        
        return container._create_instance_from_type(self.implementation)
    
    def dispose_scope(self, scope):
        """Dispose instances for a specific scope."""
        with self._lock:
            if scope in self._scoped_instances:
                instance = self._scoped_instances[scope]
                if hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing scoped {self.service_type.__name__}: {e}")
                
                del self._scoped_instances[scope]
    
    def dispose(self):
        with self._lock:
            for scope, instance in list(self._scoped_instances.items()):
                if hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing scoped {self.service_type.__name__}: {e}")
            
            self._scoped_instances.clear()


class FactoryProvider(ServiceProvider):
    """Provides instances using a custom factory function."""
    
    def __init__(self, service_type: Type[T], factory: Callable[[], T]):
        super().__init__(service_type, type(None))  # No implementation type for factory
        self.factory = factory
    
    def get_instance(self, container: 'Container', **kwargs) -> T:
        try:
            instance = self.factory()
            logger.debug(f"Created via factory: {self.service_type.__name__}")
            return instance
        except Exception as e:
            logger.error(f"Factory failed for {self.service_type.__name__}: {e}")
            raise
    
    def dispose(self):
        pass  # Factory doesn't manage instances directly


class InstanceProvider(ServiceProvider):
    """Provides a pre-created instance."""
    
    def __init__(self, service_type: Type[T], instance: T):
        super().__init__(service_type, type(instance))
        self.instance = instance
    
    def get_instance(self, container: 'Container', **kwargs) -> T:
        return self.instance
    
    def dispose(self):
        if hasattr(self.instance, 'dispose'):
            try:
                self.instance.dispose()
            except Exception as e:
                logger.warning(f"Error disposing instance {self.service_type.__name__}: {e}")


class ConditionalProvider(ServiceProvider):
    """Provides instances based on conditions."""
    
    def __init__(
        self,
        service_type: Type[T],
        condition: Callable[[], bool],
        true_provider: ServiceProvider,
        false_provider: ServiceProvider
    ):
        super().__init__(service_type, true_provider.implementation)
        self.condition = condition
        self.true_provider = true_provider
        self.false_provider = false_provider
    
    def get_instance(self, container: 'Container', **kwargs) -> T:
        if self.condition():
            return self.true_provider.get_instance(container, **kwargs)
        else:
            return self.false_provider.get_instance(container, **kwargs)
    
    def dispose(self):
        self.true_provider.dispose()
        self.false_provider.dispose()


class LazyProvider(ServiceProvider):
    """Provides lazy-loaded instances."""
    
    def __init__(self, service_type: Type[T], provider: ServiceProvider):
        super().__init__(service_type, provider.implementation)
        self.provider = provider
        self._instance: Optional[T] = None
        self._loaded = False
        self._lock = threading.Lock()
    
    def get_instance(self, container: 'Container', **kwargs) -> T:
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._instance = self.provider.get_instance(container, **kwargs)
                    self._loaded = True
                    logger.debug(f"Lazy loaded: {self.service_type.__name__}")
        
        return self._instance
    
    def dispose(self):
        with self._lock:
            self.provider.dispose()
            if self._instance and hasattr(self._instance, 'dispose'):
                try:
                    self._instance.dispose()
                except Exception as e:
                    logger.warning(f"Error disposing lazy {self.service_type.__name__}: {e}")
            
            self._instance = None
            self._loaded = False


class PooledProvider(ServiceProvider):
    """Provides instances from a pool for reuse."""
    
    def __init__(self, service_type: Type[T], implementation: Type[T], pool_size: int = 10):
        super().__init__(service_type, implementation)
        self.pool_size = pool_size
        self._available_instances: list[T] = []
        self._all_instances: set[T] = set()
        self._lock = threading.Lock()
    
    def get_instance(self, container: 'Container', **kwargs) -> T:
        with self._lock:
            if self._available_instances:
                instance = self._available_instances.pop()
                logger.debug(f"Reused pooled: {self.service_type.__name__}")
                return instance
            
            # Create new instance if pool not full
            if len(self._all_instances) < self.pool_size:
                instance = self._create_instance(container)
                self._all_instances.add(instance)
                logger.debug(f"Created pooled: {self.service_type.__name__}")
                return instance
            
            # Pool is full, wait or create temporary
            instance = self._create_instance(container)
            logger.debug(f"Created temporary (pool full): {self.service_type.__name__}")
            return instance
    
    def _create_instance(self, container: 'Container') -> T:
        """Create a pooled instance."""
        from .container import Container  # Avoid circular import
        
        if not isinstance(container, Container):
            raise ValueError("Container must be a Container instance")
        
        return container._create_instance_from_type(self.implementation)
    
    def return_instance(self, instance: T):
        """Return an instance to the pool."""
        with self._lock:
            if instance in self._all_instances and len(self._available_instances) < self.pool_size:
                # Reset instance state if possible
                if hasattr(instance, 'reset'):
                    try:
                        instance.reset()
                    except Exception as e:
                        logger.warning(f"Error resetting pooled instance: {e}")
                        return
                
                self._available_instances.append(instance)
                logger.debug(f"Returned to pool: {self.service_type.__name__}")
    
    def dispose(self):
        with self._lock:
            for instance in self._all_instances:
                if hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing pooled {self.service_type.__name__}: {e}")
            
            self._available_instances.clear()
            self._all_instances.clear()