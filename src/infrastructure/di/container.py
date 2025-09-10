"""
Core dependency injection container implementation.

Provides service registration, automatic dependency resolution,
and lifecycle management with support for singleton, transient,
and scoped services.
"""

import inspect
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, List, Type, TypeVar, Optional, Callable, Union
from weakref import WeakSet

from src.shared.exceptions import (
    TradingPlatformError,
    ValidationError
)
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class ServiceLifetime(Enum):
    """Service lifetime management options."""
    SINGLETON = "singleton"
    TRANSIENT = "transient" 
    SCOPED = "scoped"


class DependencyResolutionError(TradingPlatformError):
    """Raised when dependency resolution fails."""
    pass


class CircularDependencyError(DependencyResolutionError):
    """Raised when circular dependencies are detected."""
    pass


class ServiceNotRegisteredError(DependencyResolutionError):
    """Raised when attempting to resolve unregistered service."""
    pass


class ServiceDescriptor:
    """Describes a registered service."""
    
    def __init__(
        self,
        service_type: Type,
        implementation: Union[Type, Callable],
        lifetime: ServiceLifetime,
        factory: Optional[Callable] = None,
        dependencies: Optional[Dict[str, Type]] = None
    ):
        self.service_type = service_type
        self.implementation = implementation
        self.lifetime = lifetime
        self.factory = factory
        self.dependencies = dependencies or {}
        
    def __repr__(self) -> str:
        return (f"ServiceDescriptor(service_type={self.service_type.__name__}, "
                f"implementation={self.implementation.__name__}, "
                f"lifetime={self.lifetime.value})")


class ServiceScope:
    """Manages scoped service instances."""
    
    def __init__(self):
        self._instances: Dict[Type, Any] = {}
        self._disposed = False
        self._lock = threading.Lock()
        
    def get_or_create(self, service_type: Type, factory: Callable[[], Any]) -> Any:
        """Get existing instance or create new one."""
        if self._disposed:
            raise DependencyResolutionError("Service scope has been disposed")
            
        with self._lock:
            if service_type not in self._instances:
                self._instances[service_type] = factory()
            return self._instances[service_type]
    
    def dispose(self):
        """Dispose all scoped instances."""
        if self._disposed:
            return
            
        with self._lock:
            for instance in self._instances.values():
                if hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing service: {e}")
                        
            self._instances.clear()
            self._disposed = True


class Container:
    """Main dependency injection container."""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._lock = threading.RLock()
        self._resolution_stack: List[Type] = []
        self._scopes: WeakSet[ServiceScope] = WeakSet()
        
        # Register self
        self.register_instance(Container, self)
        
        logger.debug("Dependency injection container initialized")
    
    def register(
        self,
        service_type: Type[T],
        implementation: Union[Type[T], Callable[[], T], None] = None,
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
        factory: Optional[Callable] = None
    ) -> 'Container':
        """
        Register a service with the container.
        
        Args:
            service_type: Interface or abstract class
            implementation: Concrete implementation class
            lifetime: Service lifetime management
            factory: Custom factory function
            
        Returns:
            Container instance for method chaining
        """
        if implementation is None:
            implementation = service_type
            
        with self._lock:
            # Analyze dependencies
            dependencies = self._analyze_dependencies(implementation)
            
            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation=implementation,
                lifetime=lifetime,
                factory=factory,
                dependencies=dependencies
            )
            
            self._services[service_type] = descriptor
            
            logger.debug(f"Registered service: {service_type.__name__} -> {implementation.__name__} ({lifetime.value})")
            
        return self
    
    def register_instance(self, service_type: Type[T], instance: T) -> 'Container':
        """Register a specific instance as a singleton."""
        with self._lock:
            self._services[service_type] = ServiceDescriptor(
                service_type=service_type,
                implementation=type(instance),
                lifetime=ServiceLifetime.SINGLETON
            )
            self._singletons[service_type] = instance
            
            logger.debug(f"Registered instance: {service_type.__name__}")
            
        return self
    
    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[[], T],
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT
    ) -> 'Container':
        """Register a factory function for service creation."""
        return self.register(service_type, None, lifetime, factory)
    
    def resolve(self, service_type: Type[T], scope: Optional[ServiceScope] = None) -> T:
        """
        Resolve a service instance.
        
        Args:
            service_type: Type of service to resolve
            scope: Optional service scope for scoped services
            
        Returns:
            Service instance
            
        Raises:
            ServiceNotRegisteredError: If service not registered
            CircularDependencyError: If circular dependency detected
        """
        with self._lock:
            # Check for circular dependencies
            if service_type in self._resolution_stack:
                cycle = ' -> '.join(t.__name__ for t in self._resolution_stack[self._resolution_stack.index(service_type):])
                raise CircularDependencyError(f"Circular dependency detected: {cycle} -> {service_type.__name__}")
            
            if service_type not in self._services:
                raise ServiceNotRegisteredError(f"Service {service_type.__name__} is not registered")
            
            descriptor = self._services[service_type]
            
            # Handle different lifetimes
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                return self._resolve_singleton(descriptor)
            elif descriptor.lifetime == ServiceLifetime.TRANSIENT:
                return self._resolve_transient(descriptor)
            elif descriptor.lifetime == ServiceLifetime.SCOPED:
                if scope is None:
                    raise DependencyResolutionError("Scoped service requires a service scope")
                return self._resolve_scoped(descriptor, scope)
            
            raise DependencyResolutionError(f"Unknown service lifetime: {descriptor.lifetime}")
    
    def _resolve_singleton(self, descriptor: ServiceDescriptor) -> Any:
        """Resolve singleton service."""
        if descriptor.service_type in self._singletons:
            return self._singletons[descriptor.service_type]
        
        # Create singleton instance
        instance = self._create_instance(descriptor)
        self._singletons[descriptor.service_type] = instance
        
        logger.debug(f"Created singleton instance: {descriptor.service_type.__name__}")
        return instance
    
    def _resolve_transient(self, descriptor: ServiceDescriptor) -> Any:
        """Resolve transient service (new instance each time)."""
        instance = self._create_instance(descriptor)
        logger.debug(f"Created transient instance: {descriptor.service_type.__name__}")
        return instance
    
    def _resolve_scoped(self, descriptor: ServiceDescriptor, scope: ServiceScope) -> Any:
        """Resolve scoped service (one instance per scope)."""
        return scope.get_or_create(
            descriptor.service_type,
            lambda: self._create_instance(descriptor)
        )
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create service instance with dependency injection."""
        self._resolution_stack.append(descriptor.service_type)
        
        try:
            if descriptor.factory:
                # Use custom factory
                return descriptor.factory()
            
            # Resolve constructor dependencies
            dependencies = {}
            for param_name, param_type in descriptor.dependencies.items():
                dependencies[param_name] = self.resolve(param_type)
            
            # Create instance
            instance = descriptor.implementation(**dependencies)
            
            logger.debug(f"Created instance: {descriptor.implementation.__name__} with dependencies: {list(dependencies.keys())}")
            return instance
            
        finally:
            self._resolution_stack.pop()
    
    def _create_instance_from_type(self, implementation_type: Type) -> Any:
        """Create instance from type with automatic dependency resolution."""
        # Analyze dependencies
        dependencies = self._analyze_dependencies(implementation_type)
        
        # Resolve dependencies
        resolved_dependencies = {}
        for param_name, param_type in dependencies.items():
            resolved_dependencies[param_name] = self.resolve(param_type)
        
        # Create instance
        instance = implementation_type(**resolved_dependencies)
        logger.debug(f"Created instance from type: {implementation_type.__name__}")
        return instance
    
    def _analyze_dependencies(self, implementation: Type) -> Dict[str, Type]:
        """Analyze constructor dependencies using type hints."""
        if not inspect.isclass(implementation):
            return {}
        
        try:
            signature = inspect.signature(implementation.__init__)
            dependencies = {}
            
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                    
                if param.annotation != inspect.Parameter.empty:
                    dependencies[param_name] = param.annotation
                else:
                    logger.warning(f"No type annotation for parameter '{param_name}' in {implementation.__name__}")
            
            return dependencies
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not analyze dependencies for {implementation.__name__}: {e}")
            return {}
    
    @contextmanager
    def scope(self):
        """Create a new service scope."""
        service_scope = ServiceScope()
        self._scopes.add(service_scope)
        
        try:
            yield service_scope
        finally:
            service_scope.dispose()
    
    def injectable(self, cls: Type[T]) -> Type[T]:
        """Decorator to mark a class as injectable."""
        # Auto-register the class
        self.register(cls, cls)
        return cls
    
    def dispose(self):
        """Dispose all singleton instances and scopes."""
        with self._lock:
            # Dispose singletons
            for instance in self._singletons.values():
                if hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing singleton: {e}")
            
            # Clear collections
            self._singletons.clear()
            self._services.clear()
            
            # Dispose remaining scopes
            for scope in list(self._scopes):
                scope.dispose()
                
        logger.debug("Container disposed")
    
    def get_registered_services(self) -> Dict[Type, ServiceDescriptor]:
        """Get all registered services for debugging."""
        with self._lock:
            return self._services.copy()


# Global container instance
_default_container: Optional[Container] = None
_container_lock = threading.Lock()


def get_container() -> Container:
    """Get the default container instance."""
    global _default_container
    
    if _default_container is None:
        with _container_lock:
            if _default_container is None:
                _default_container = Container()
                
    return _default_container


def set_container(container: Container):
    """Set the default container instance."""
    global _default_container
    
    with _container_lock:
        _default_container = container