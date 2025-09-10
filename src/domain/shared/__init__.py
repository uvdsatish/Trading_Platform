"""
Shared domain concepts and base classes.

This module provides the foundational building blocks for all domain models
following Domain-Driven Design patterns.
"""

from .entities import Entity, AggregateRoot
from .value_objects import ValueObject
from .events import DomainEvent, DomainEventHandler
from .services import DomainService
from .repositories import Repository, Specification
from .identifiers import EntityId, AggregateId

__all__ = [
    # Base domain building blocks
    'Entity',
    'AggregateRoot', 
    'ValueObject',
    'DomainEvent',
    'DomainEventHandler',
    'DomainService',
    'Repository',
    'Specification',
    'EntityId',
    'AggregateId',
]