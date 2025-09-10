"""
Base value object implementations.
"""

from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ValueObject(ABC):
    """
    Base class for all value objects.
    
    Value objects have no conceptual identity and are defined by their attributes.
    They are immutable and have structural equality.
    """
    
    @abstractmethod
    def validate(self) -> None:
        """Validate the value object's constraints."""
        pass
    
    def __post_init__(self):
        self.validate()


@dataclass(frozen=True)
class Money(ValueObject):
    """Represents a monetary amount with currency."""
    
    amount: Decimal
    currency: str = "USD"
    
    def validate(self) -> None:
        if self.amount is None:
            raise ValueError("Amount cannot be None")
        if not isinstance(self.amount, Decimal):
            raise ValueError("Amount must be a Decimal")
        if len(self.currency) != 3:
            raise ValueError("Currency must be a 3-letter code")
    
    def add(self, other: 'Money') -> 'Money':
        """Add two money amounts (must be same currency)."""
        if self.currency != other.currency:
            raise ValueError(f"Cannot add {self.currency} and {other.currency}")
        return Money(self.amount + other.amount, self.currency)
    
    def subtract(self, other: 'Money') -> 'Money':
        """Subtract two money amounts (must be same currency)."""
        if self.currency != other.currency:
            raise ValueError(f"Cannot subtract {other.currency} from {self.currency}")
        return Money(self.amount - other.amount, self.currency)
    
    def multiply(self, factor: Decimal) -> 'Money':
        """Multiply money by a factor."""
        return Money(self.amount * factor, self.currency)
    
    def is_positive(self) -> bool:
        """Check if amount is positive."""
        return self.amount > 0
    
    def is_negative(self) -> bool:
        """Check if amount is negative."""
        return self.amount < 0
    
    def is_zero(self) -> bool:
        """Check if amount is zero."""
        return self.amount == 0


@dataclass(frozen=True)
class Price(ValueObject):
    """Represents a financial instrument price."""
    
    value: Decimal
    
    def validate(self) -> None:
        if self.value is None:
            raise ValueError("Price value cannot be None")
        if not isinstance(self.value, Decimal):
            raise ValueError("Price value must be a Decimal")
        if self.value < 0:
            raise ValueError("Price cannot be negative")
    
    def to_money(self, currency: str = "USD") -> Money:
        """Convert price to money."""
        return Money(self.value, currency)


@dataclass(frozen=True)
class Quantity(ValueObject):
    """Represents a quantity of shares or units."""
    
    value: int
    
    def validate(self) -> None:
        if self.value is None:
            raise ValueError("Quantity value cannot be None")
        if not isinstance(self.value, int):
            raise ValueError("Quantity value must be an integer")
        if self.value < 0:
            raise ValueError("Quantity cannot be negative")
    
    def add(self, other: 'Quantity') -> 'Quantity':
        """Add two quantities."""
        return Quantity(self.value + other.value)
    
    def subtract(self, other: 'Quantity') -> 'Quantity':
        """Subtract two quantities."""
        return Quantity(self.value - other.value)
    
    def is_zero(self) -> bool:
        """Check if quantity is zero."""
        return self.value == 0


@dataclass(frozen=True)
class Percentage(ValueObject):
    """Represents a percentage value."""
    
    value: Decimal
    
    def validate(self) -> None:
        if self.value is None:
            raise ValueError("Percentage value cannot be None")
        if not isinstance(self.value, Decimal):
            raise ValueError("Percentage value must be a Decimal")
    
    def to_decimal(self) -> Decimal:
        """Convert percentage to decimal (e.g., 50% -> 0.5)."""
        return self.value / 100
    
    def apply_to(self, amount: Decimal) -> Decimal:
        """Apply percentage to an amount."""
        return amount * self.to_decimal()


@dataclass(frozen=True)
class DateRange(ValueObject):
    """Represents a date range."""
    
    start_date: Any  # Should be datetime.date but avoiding import
    end_date: Any
    
    def validate(self) -> None:
        if self.start_date is None or self.end_date is None:
            raise ValueError("Start and end dates cannot be None")
        if self.start_date > self.end_date:
            raise ValueError("Start date must be before or equal to end date")
    
    def contains(self, date: Any) -> bool:
        """Check if date is within range."""
        return self.start_date <= date <= self.end_date
    
    def duration_days(self) -> int:
        """Get duration in days."""
        return (self.end_date - self.start_date).days