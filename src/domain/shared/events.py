"""
Domain events and event handling.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Callable
from datetime import datetime
from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass(frozen=True)
class DomainEvent(ABC):
    """
    Base class for all domain events.
    
    Domain events represent something that happened in the domain that is
    of interest to other parts of the system.
    """
    
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    aggregate_id: Any = field(default=None)
    version: int = field(default=1)
    
    @abstractmethod
    def get_event_type(self) -> str:
        """Get the event type identifier."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            'event_id': str(self.event_id),
            'event_type': self.get_event_type(),
            'occurred_at': self.occurred_at.isoformat(),
            'aggregate_id': str(self.aggregate_id) if self.aggregate_id else None,
            'version': self.version,
            'data': self._get_event_data()
        }
    
    @abstractmethod
    def _get_event_data(self) -> Dict[str, Any]:
        """Get event-specific data."""
        pass


class DomainEventHandler(ABC):
    """Base class for domain event handlers."""
    
    @abstractmethod
    def handle(self, event: DomainEvent) -> None:
        """Handle a domain event."""
        pass
    
    @abstractmethod
    def can_handle(self, event_type: str) -> bool:
        """Check if this handler can handle the event type."""
        pass


class DomainEventPublisher:
    """Publisher for domain events."""
    
    def __init__(self):
        self._handlers: Dict[str, List[DomainEventHandler]] = {}
        self._global_handlers: List[DomainEventHandler] = []
    
    def subscribe(self, event_type: str, handler: DomainEventHandler) -> None:
        """Subscribe a handler to a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def subscribe_global(self, handler: DomainEventHandler) -> None:
        """Subscribe a handler to all events."""
        self._global_handlers.append(handler)
    
    def publish(self, event: DomainEvent) -> None:
        """Publish a domain event to all interested handlers."""
        event_type = event.get_event_type()
        
        # Handle specific event type handlers
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    handler.handle(event)
                except Exception as e:
                    # Log error but don't break other handlers
                    print(f"Error handling event {event_type}: {e}")
        
        # Handle global handlers
        for handler in self._global_handlers:
            try:
                if handler.can_handle(event_type):
                    handler.handle(event)
            except Exception as e:
                # Log error but don't break other handlers
                print(f"Error in global handler for event {event_type}: {e}")
    
    def publish_all(self, events: List[DomainEvent]) -> None:
        """Publish multiple events."""
        for event in events:
            self.publish(event)


# Global event publisher instance
_event_publisher = DomainEventPublisher()


def get_event_publisher() -> DomainEventPublisher:
    """Get the global event publisher instance."""
    return _event_publisher


# Market Data Events
@dataclass(frozen=True)
class MarketDataUpdated(DomainEvent):
    """Event raised when market data is updated."""
    
    ticker: str
    data_type: str  # 'ohlcv', 'quote', etc.
    
    def get_event_type(self) -> str:
        return "market_data.updated"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'data_type': self.data_type
        }


# Trading Events
@dataclass(frozen=True)
class OrderPlaced(DomainEvent):
    """Event raised when an order is placed."""
    
    order_id: str
    ticker: str
    side: str
    quantity: int
    price: Any
    
    def get_event_type(self) -> str:
        return "trading.order_placed"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'ticker': self.ticker,
            'side': self.side,
            'quantity': self.quantity,
            'price': str(self.price)
        }


@dataclass(frozen=True)
class OrderFilled(DomainEvent):
    """Event raised when an order is filled."""
    
    order_id: str
    fill_price: Any
    fill_quantity: int
    
    def get_event_type(self) -> str:
        return "trading.order_filled"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'fill_price': str(self.fill_price),
            'fill_quantity': self.fill_quantity
        }


# Technical Analysis Events
@dataclass(frozen=True)
class SignalGenerated(DomainEvent):
    """Event raised when a trading signal is generated."""
    
    ticker: str
    signal_type: str
    strength: float
    strategy: str
    
    def get_event_type(self) -> str:
        return "technical_analysis.signal_generated"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'signal_type': self.signal_type,
            'strength': self.strength,
            'strategy': self.strategy
        }