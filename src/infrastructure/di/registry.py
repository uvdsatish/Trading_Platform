"""
Service registry for dependency injection container.

Manages service registrations, provides lookup capabilities,
and handles service lifecycle coordination.
"""

import threading
from collections import defaultdict
from typing import Dict, List, Type, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from src.infrastructure.logging import get_logger
from .providers import ServiceProvider

logger = get_logger(__name__)


@dataclass
class ServiceRegistration:
    """Registration information for a service."""
    service_type: Type
    provider: ServiceProvider
    registration_time: datetime = field(default_factory=datetime.now)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
        if self.metadata is None:
            self.metadata = {}


class ServiceRegistry:
    """Registry for managing service registrations."""
    
    def __init__(self):
        self._registrations: Dict[Type, ServiceRegistration] = {}
        self._by_tag: Dict[str, Set[Type]] = defaultdict(set)
        self._dependencies: Dict[Type, Set[Type]] = defaultdict(set)
        self._dependents: Dict[Type, Set[Type]] = defaultdict(set)
        self._lock = threading.RLock()
        
        logger.debug("Service registry initialized")
    
    def register(
        self,
        service_type: Type,
        provider: ServiceProvider,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a service with the registry.
        
        Args:
            service_type: The service interface/type
            provider: Service provider for instantiation
            tags: Optional tags for categorization
            metadata: Optional metadata
        """
        with self._lock:
            registration = ServiceRegistration(
                service_type=service_type,
                provider=provider,
                tags=tags or set(),
                metadata=metadata or {}
            )
            
            # Remove old registration if exists
            if service_type in self._registrations:
                self._unregister_internal(service_type)
            
            # Add new registration
            self._registrations[service_type] = registration
            
            # Update tag index
            for tag in registration.tags:
                self._by_tag[tag].add(service_type)
            
            logger.debug(f"Registered service: {service_type.__name__} with tags: {registration.tags}")
    
    def unregister(self, service_type: Type) -> bool:
        """
        Unregister a service.
        
        Args:
            service_type: Service type to unregister
            
        Returns:
            True if service was unregistered, False if not found
        """
        with self._lock:
            return self._unregister_internal(service_type)
    
    def _unregister_internal(self, service_type: Type) -> bool:
        """Internal unregistration logic."""
        if service_type not in self._registrations:
            return False
        
        registration = self._registrations[service_type]
        
        # Remove from tag index
        for tag in registration.tags:
            self._by_tag[tag].discard(service_type)
            if not self._by_tag[tag]:
                del self._by_tag[tag]
        
        # Clean up dependency tracking
        dependencies = self._dependencies.get(service_type, set())
        for dep in dependencies:
            self._dependents[dep].discard(service_type)
        
        dependents = self._dependents.get(service_type, set())
        for dependent in dependents:
            self._dependencies[dependent].discard(service_type)
        
        # Remove registration
        del self._registrations[service_type]
        
        # Dispose provider
        try:
            registration.provider.dispose()
        except Exception as e:
            logger.warning(f"Error disposing provider for {service_type.__name__}: {e}")
        
        logger.debug(f"Unregistered service: {service_type.__name__}")
        return True
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        with self._lock:
            return service_type in self._registrations
    
    def get_registration(self, service_type: Type) -> Optional[ServiceRegistration]:
        """Get registration for a service type."""
        with self._lock:
            return self._registrations.get(service_type)
    
    def get_provider(self, service_type: Type) -> Optional[ServiceProvider]:
        """Get provider for a service type."""
        with self._lock:
            registration = self._registrations.get(service_type)
            return registration.provider if registration else None
    
    def get_services_by_tag(self, tag: str) -> List[Type]:
        """Get all service types with a specific tag."""
        with self._lock:
            return list(self._by_tag.get(tag, set()))
    
    def get_all_services(self) -> List[Type]:
        """Get all registered service types."""
        with self._lock:
            return list(self._registrations.keys())
    
    def get_registrations_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about all registrations."""
        with self._lock:
            info = []
            for service_type, registration in self._registrations.items():
                info.append({
                    'service_type': service_type.__name__,
                    'provider_type': type(registration.provider).__name__,
                    'registration_time': registration.registration_time,
                    'tags': list(registration.tags),
                    'metadata': registration.metadata.copy(),
                    'dependencies': [dep.__name__ for dep in self._dependencies.get(service_type, set())],
                    'dependents': [dep.__name__ for dep in self._dependents.get(service_type, set())]
                })
            return info
    
    def track_dependency(self, service_type: Type, dependency_type: Type) -> None:
        """Track a dependency relationship."""
        with self._lock:
            self._dependencies[service_type].add(dependency_type)
            self._dependents[dependency_type].add(service_type)
    
    def get_dependencies(self, service_type: Type) -> Set[Type]:
        """Get direct dependencies of a service."""
        with self._lock:
            return self._dependencies.get(service_type, set()).copy()
    
    def get_dependents(self, service_type: Type) -> Set[Type]:
        """Get services that depend on the given service."""
        with self._lock:
            return self._dependents.get(service_type, set()).copy()
    
    def get_dependency_chain(self, service_type: Type) -> List[Type]:
        """Get the full dependency chain for a service (topological order)."""
        with self._lock:
            visited = set()
            chain = []
            
            def visit(current_type: Type):
                if current_type in visited:
                    return
                
                visited.add(current_type)
                
                # Visit dependencies first
                for dep in self._dependencies.get(current_type, set()):
                    visit(dep)
                
                chain.append(current_type)
            
            visit(service_type)
            return chain
    
    def detect_circular_dependencies(self) -> List[List[Type]]:
        """Detect circular dependencies in the registry."""
        with self._lock:
            cycles = []
            visited = set()
            rec_stack = set()
            path = []
            
            def visit(service_type: Type) -> bool:
                if service_type in rec_stack:
                    # Found cycle
                    cycle_start = path.index(service_type)
                    cycle = path[cycle_start:] + [service_type]
                    cycles.append(cycle)
                    return True
                
                if service_type in visited:
                    return False
                
                visited.add(service_type)
                rec_stack.add(service_type)
                path.append(service_type)
                
                # Check dependencies
                for dep in self._dependencies.get(service_type, set()):
                    if visit(dep):
                        return True
                
                rec_stack.remove(service_type)
                path.pop()
                return False
            
            # Check all services
            for service_type in self._registrations:
                if service_type not in visited:
                    visit(service_type)
            
            return cycles
    
    def validate_registry(self) -> List[str]:
        """Validate the registry and return any issues."""
        issues = []
        
        with self._lock:
            # Check for circular dependencies
            cycles = self.detect_circular_dependencies()
            for cycle in cycles:
                cycle_str = ' -> '.join(t.__name__ for t in cycle)
                issues.append(f"Circular dependency: {cycle_str}")
            
            # Check for missing dependencies
            for service_type, registration in self._registrations.items():
                dependencies = self._dependencies.get(service_type, set())
                for dep in dependencies:
                    if dep not in self._registrations:
                        issues.append(f"Missing dependency: {service_type.__name__} depends on {dep.__name__} which is not registered")
            
            # Check for orphaned services
            root_services = []
            for service_type in self._registrations:
                if not self._dependents.get(service_type):
                    root_services.append(service_type.__name__)
            
            if len(root_services) > 20:  # Arbitrary threshold
                issues.append(f"Many root services ({len(root_services)}), consider reviewing architecture")
        
        return issues
    
    def clear(self) -> None:
        """Clear all registrations."""
        with self._lock:
            # Dispose all providers
            for registration in self._registrations.values():
                try:
                    registration.provider.dispose()
                except Exception as e:
                    logger.warning(f"Error disposing provider: {e}")
            
            # Clear all collections
            self._registrations.clear()
            self._by_tag.clear()
            self._dependencies.clear()
            self._dependents.clear()
            
        logger.debug("Service registry cleared")
    
    def clone(self) -> 'ServiceRegistry':
        """Create a copy of this registry."""
        with self._lock:
            new_registry = ServiceRegistry()
            
            # Copy registrations (providers are shared, not cloned)
            for service_type, registration in self._registrations.items():
                new_registry.register(
                    service_type,
                    registration.provider,
                    registration.tags.copy(),
                    registration.metadata.copy()
                )
            
            # Copy dependency tracking
            new_registry._dependencies = {
                k: v.copy() for k, v in self._dependencies.items()
            }
            new_registry._dependents = {
                k: v.copy() for k, v in self._dependents.items()
            }
            
            return new_registry
    
    def merge(self, other: 'ServiceRegistry', overwrite: bool = False) -> None:
        """
        Merge another registry into this one.
        
        Args:
            other: Registry to merge
            overwrite: Whether to overwrite existing registrations
        """
        with self._lock:
            for service_type, registration in other._registrations.items():
                if service_type not in self._registrations or overwrite:
                    self.register(
                        service_type,
                        registration.provider,
                        registration.tags.copy(),
                        registration.metadata.copy()
                    )
                    
                    # Copy dependency tracking
                    if service_type in other._dependencies:
                        self._dependencies[service_type] = other._dependencies[service_type].copy()
                    
                    if service_type in other._dependents:
                        self._dependents[service_type] = other._dependents[service_type].copy()
        
        logger.debug(f"Merged registry with {len(other._registrations)} services")