"""
Base entity and aggregate root implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from .identifiers import EntityId, AggregateId
from .events import DomainEvent


@dataclass
class Entity(ABC):
    """
    Base class for all domain entities.
    
    An entity is an object that has a distinct identity that runs through time
    and different representations. Entities are mutable and have identity equality.
    """
    
    id: EntityId
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = field(default=None)
    version: int = field(default=1)
    
    def __post_init__(self):
        if self.id is None:
            raise ValueError("Entity ID cannot be None")
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        return hash((self.__class__, self.id))
    
    def mark_updated(self) -> None:
        """Mark the entity as updated."""
        self.updated_at = datetime.utcnow()
        self.version += 1
    
    @abstractmethod
    def validate(self) -> None:
        """Validate the entity's business rules."""
        pass


@dataclass
class AggregateRoot(Entity):
    """
    Base class for aggregate roots.
    
    An aggregate root is the only member of its aggregate that objects outside
    the aggregate are allowed to hold references to. It controls access to the
    aggregate and ensures invariants are maintained.
    """
    
    _domain_events: List[DomainEvent] = field(default_factory=list, init=False)
    
    def add_domain_event(self, event: DomainEvent) -> None:
        """Add a domain event to be published."""
        self._domain_events.append(event)
    
    def clear_domain_events(self) -> None:
        """Clear all domain events."""
        self._domain_events.clear()
    
    def get_domain_events(self) -> List[DomainEvent]:
        """Get all pending domain events."""
        return self._domain_events.copy()
    
    def has_domain_events(self) -> bool:
        """Check if there are pending domain events."""
        return len(self._domain_events) > 0