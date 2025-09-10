"""
Market data value objects.
"""

from dataclasses import dataclass
from decimal import Decimal
from datetime import date, time, datetime
from typing import Optional

from ..shared.value_objects import ValueObject, Price, Quantity


@dataclass(frozen=True)
class Volume(ValueObject):
    """Represents trading volume."""
    
    value: int
    
    def validate(self) -> None:
        if self.value is None:
            raise ValueError("Volume cannot be None")
        if not isinstance(self.value, int):
            raise ValueError("Volume must be an integer")
        if self.value < 0:
            raise ValueError("Volume cannot be negative")
    
    def add(self, other: 'Volume') -> 'Volume':
        """Add two volumes."""
        return Volume(self.value + other.value)
    
    def is_zero(self) -> bool:
        """Check if volume is zero."""
        return self.value == 0


@dataclass(frozen=True)
class OHLCV_Data(ValueObject):
    """OHLCV (Open, High, Low, Close, Volume) data."""
    
    open_price: Price
    high_price: Price
    low_price: Price
    close_price: Price
    volume: Volume
    
    def validate(self) -> None:
        if any(x is None for x in [self.open_price, self.high_price, self.low_price, self.close_price, self.volume]):
            raise ValueError("All OHLCV components must be provided")
        
        # Validate price relationships
        if self.high_price.value < max(self.open_price.value, self.close_price.value):
            raise ValueError("High price must be >= open and close prices")
        
        if self.low_price.value > min(self.open_price.value, self.close_price.value):
            raise ValueError("Low price must be <= open and close prices")
        
        if self.high_price.value < self.low_price.value:
            raise ValueError("High price must be >= low price")
    
    def get_typical_price(self) -> Price:
        """Calculate typical price (HLC/3)."""
        typical = (self.high_price.value + self.low_price.value + self.close_price.value) / 3
        return Price(typical)
    
    def get_range(self) -> Price:
        """Get the high-low range."""
        return Price(self.high_price.value - self.low_price.value)
    
    def is_up_day(self) -> bool:
        """Check if close > open."""
        return self.close_price.value > self.open_price.value
    
    def is_down_day(self) -> bool:
        """Check if close < open."""
        return self.close_price.value < self.open_price.value
    
    def is_doji(self, threshold: Decimal = Decimal('0.01')) -> bool:
        """Check if it's a doji (open ≈ close)."""
        body_size = abs(self.close_price.value - self.open_price.value)
        return body_size <= threshold


@dataclass(frozen=True)
class QuoteData(ValueObject):
    """Bid/Ask quote data."""
    
    bid_price: Price
    ask_price: Price
    bid_size: Quantity
    ask_size: Quantity
    
    def validate(self) -> None:
        if any(x is None for x in [self.bid_price, self.ask_price, self.bid_size, self.ask_size]):
            raise ValueError("All quote components must be provided")
        
        if self.bid_price.value >= self.ask_price.value:
            raise ValueError("Bid price must be less than ask price")
    
    def get_spread(self) -> 'Spread':
        """Get the bid-ask spread."""
        spread_value = self.ask_price.value - self.bid_price.value
        return Spread(spread_value)
    
    def get_mid_price(self) -> Price:
        """Get the mid price."""
        mid = (self.bid_price.value + self.ask_price.value) / 2
        return Price(mid)


@dataclass(frozen=True)
class Spread(ValueObject):
    """Bid-ask spread."""
    
    value: Decimal
    
    def validate(self) -> None:
        if self.value is None:
            raise ValueError("Spread value cannot be None")
        if not isinstance(self.value, Decimal):
            raise ValueError("Spread must be a Decimal")
        if self.value < 0:
            raise ValueError("Spread cannot be negative")
    
    def to_percentage(self, mid_price: Price) -> Decimal:
        """Convert spread to percentage of mid price."""
        if mid_price.value == 0:
            return Decimal('0')
        return (self.value / mid_price.value) * 100


@dataclass(frozen=True)
class MarketTime(ValueObject):
    """Market time with timezone awareness."""
    
    time_value: time
    timezone: str = "US/Eastern"
    
    def validate(self) -> None:
        if self.time_value is None:
            raise ValueError("Time value cannot be None")
        if not self.timezone:
            raise ValueError("Timezone must be specified")
    
    def is_market_hours(self) -> bool:
        """Check if time falls within regular market hours (9:30 AM - 4:00 PM ET)."""
        market_open = time(9, 30)
        market_close = time(16, 0)
        return market_open <= self.time_value <= market_close
    
    def is_pre_market(self) -> bool:
        """Check if time is pre-market (4:00 AM - 9:30 AM ET)."""
        pre_market_start = time(4, 0)
        market_open = time(9, 30)
        return pre_market_start <= self.time_value < market_open
    
    def is_after_hours(self) -> bool:
        """Check if time is after-hours (4:00 PM - 8:00 PM ET)."""
        market_close = time(16, 0)
        after_hours_end = time(20, 0)
        return market_close < self.time_value <= after_hours_end


@dataclass(frozen=True)
class TradingDay(ValueObject):
    """Represents a trading day."""
    
    date_value: date
    is_trading_day: bool = True
    
    def validate(self) -> None:
        if self.date_value is None:
            raise ValueError("Date value cannot be None")
    
    def is_weekend(self) -> bool:
        """Check if date falls on weekend."""
        return self.date_value.weekday() >= 5  # Saturday = 5, Sunday = 6
    
    def is_holiday(self) -> bool:
        """Check if date is a market holiday."""
        # This would typically integrate with a holiday calendar service
        # For now, just return False
        return False
    
    def next_trading_day(self) -> 'TradingDay':
        """Get the next trading day."""
        from datetime import timedelta
        next_date = self.date_value + timedelta(days=1)
        
        # Skip weekends (simplified logic)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        
        return TradingDay(next_date)
    
    def previous_trading_day(self) -> 'TradingDay':
        """Get the previous trading day."""
        from datetime import timedelta
        prev_date = self.date_value - timedelta(days=1)
        
        # Skip weekends (simplified logic)
        while prev_date.weekday() >= 5:
            prev_date -= timedelta(days=1)
        
        return TradingDay(prev_date)