"""
Domain identifiers and identity management.
"""

from abc import ABC
from typing import Any, Union
from uuid import UUID, uuid4
from dataclasses import dataclass


@dataclass(frozen=True)
class EntityId:
    """Base class for entity identifiers."""
    
    value: Union[str, int, UUID]
    
    def __post_init__(self):
        if self.value is None:
            raise ValueError("Entity ID cannot be None")
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        return hash((self.__class__, self.value))


@dataclass(frozen=True) 
class AggregateId(EntityId):
    """Base class for aggregate root identifiers."""
    pass


class UUIDGenerator:
    """Generates UUID-based identifiers."""
    
    @staticmethod
    def generate() -> UUID:
        return uuid4()
    
    @staticmethod
    def generate_string() -> str:
        return str(uuid4())


# Specific domain identifiers
@dataclass(frozen=True)
class TickerId(EntityId):
    """Identifier for ticker symbols."""
    
    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.value, str):
            raise ValueError("Ticker ID must be a string")
        if not self.value.strip():
            raise ValueError("Ticker ID cannot be empty")


@dataclass(frozen=True)
class OrderId(AggregateId):
    """Identifier for trading orders."""
    pass


@dataclass(frozen=True)
class PositionId(AggregateId):
    """Identifier for trading positions."""
    pass


@dataclass(frozen=True)
class PortfolioId(AggregateId):
    """Identifier for portfolios."""
    pass


@dataclass(frozen=True)
class StrategyId(EntityId):
    """Identifier for trading strategies."""
    pass


@dataclass(frozen=True)
class BacktestId(AggregateId):
    """Identifier for backtest runs."""
    pass