"""
Market data entities.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, List, Dict, Any

from ..shared.entities import Entity, AggregateRoot
from ..shared.identifiers import TickerId, EntityId
from ..shared.events import MarketDataUpdated
from .value_objects import OHLCV_Data, QuoteData, TradingDay, MarketTime


@dataclass
class Ticker(Entity):
    """
    Represents a financial instrument ticker/symbol.
    
    This is the core entity for any tradable instrument in the system.
    """
    
    id: TickerId
    symbol: str
    company_name: Optional[str] = None
    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[str] = None  # Small, Mid, Large
    is_active: bool = True
    listing_date: Optional[date] = None
    delisting_date: Optional[date] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.symbol != self.id.value:
            raise ValueError("Ticker symbol must match ID value")
    
    def validate(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("Ticker symbol cannot be empty")
        
        if self.symbol != self.symbol.upper():
            raise ValueError("Ticker symbol must be uppercase")
        
        if self.delisting_date and self.listing_date:
            if self.delisting_date <= self.listing_date:
                raise ValueError("Delisting date must be after listing date")
    
    def is_listed(self, as_of_date: date = None) -> bool:
        """Check if ticker is listed as of a specific date."""
        check_date = as_of_date or date.today()
        
        if not self.is_active:
            return False
        
        if self.listing_date and check_date < self.listing_date:
            return False
        
        if self.delisting_date and check_date >= self.delisting_date:
            return False
        
        return True
    
    def delist(self, delisting_date: date) -> None:
        """Mark ticker as delisted."""
        self.delisting_date = delisting_date
        self.is_active = False
        self.mark_updated()


@dataclass
class OHLCV(AggregateRoot):
    """
    OHLCV (Open, High, Low, Close, Volume) data entity.
    
    Represents daily market data for a specific ticker and date.
    """
    
    id: EntityId
    ticker_id: TickerId
    trading_day: TradingDay
    data: OHLCV_Data
    adjusted_close: Optional[Any] = None  # Adjusted for splits/dividends
    split_ratio: Optional[float] = None
    dividend_amount: Optional[Any] = None
    
    def validate(self) -> None:
        if not self.ticker_id:
            raise ValueError("Ticker ID is required")
        
        if not self.trading_day:
            raise ValueError("Trading day is required")
        
        if not self.data:
            raise ValueError("OHLCV data is required")
        
        if self.split_ratio and self.split_ratio <= 0:
            raise ValueError("Split ratio must be positive")
    
    def update_data(self, new_data: OHLCV_Data) -> None:
        """Update OHLCV data and raise domain event."""
        old_data = self.data
        self.data = new_data
        self.mark_updated()
        
        # Raise domain event
        event = MarketDataUpdated(
            aggregate_id=str(self.id.value),
            ticker=str(self.ticker_id.value),
            data_type='ohlcv'
        )
        self.add_domain_event(event)
    
    def apply_split(self, split_ratio: float) -> None:
        """Apply stock split to OHLCV data."""
        if split_ratio <= 0:
            raise ValueError("Split ratio must be positive")
        
        # Adjust prices (divide by split ratio)
        adjusted_open = self.data.open_price.value / split_ratio
        adjusted_high = self.data.high_price.value / split_ratio
        adjusted_low = self.data.low_price.value / split_ratio
        adjusted_close = self.data.close_price.value / split_ratio
        
        # Adjust volume (multiply by split ratio)
        adjusted_volume = int(self.data.volume.value * split_ratio)
        
        from ..shared.value_objects import Price
        from .value_objects import Volume
        
        self.data = OHLCV_Data(
            open_price=Price(adjusted_open),
            high_price=Price(adjusted_high),
            low_price=Price(adjusted_low),
            close_price=Price(adjusted_close),
            volume=Volume(adjusted_volume)
        )
        
        self.split_ratio = split_ratio
        self.mark_updated()
    
    def get_price_change(self) -> Any:
        """Get price change (close - open)."""
        return self.data.close_price.value - self.data.open_price.value
    
    def get_price_change_percent(self) -> float:
        """Get price change percentage."""
        if self.data.open_price.value == 0:
            return 0.0
        return float((self.get_price_change() / self.data.open_price.value) * 100)


@dataclass
class Quote(Entity):
    """
    Real-time or delayed quote data.
    
    Represents current bid/ask information for a ticker.
    """
    
    id: EntityId
    ticker_id: TickerId
    quote_data: QuoteData
    timestamp: datetime
    is_real_time: bool = False
    delay_minutes: int = 15
    
    def validate(self) -> None:
        if not self.ticker_id:
            raise ValueError("Ticker ID is required")
        
        if not self.quote_data:
            raise ValueError("Quote data is required")
        
        if not self.timestamp:
            raise ValueError("Timestamp is required")
        
        if self.delay_minutes < 0:
            raise ValueError("Delay minutes cannot be negative")
    
    def update_quote(self, new_quote_data: QuoteData, timestamp: datetime = None) -> None:
        """Update quote data and raise domain event."""
        self.quote_data = new_quote_data
        self.timestamp = timestamp or datetime.utcnow()
        self.mark_updated()
        
        # Raise domain event
        event = MarketDataUpdated(
            aggregate_id=str(self.id.value),
            ticker=str(self.ticker_id.value),
            data_type='quote'
        )
        self.add_domain_event(event)
    
    def is_stale(self, max_age_minutes: int = 5) -> bool:
        """Check if quote is stale based on timestamp."""
        age_minutes = (datetime.utcnow() - self.timestamp).total_seconds() / 60
        return age_minutes > max_age_minutes
    
    def get_effective_timestamp(self) -> datetime:
        """Get the effective timestamp accounting for delay."""
        if self.is_real_time:
            return self.timestamp
        
        from datetime import timedelta
        return self.timestamp - timedelta(minutes=self.delay_minutes)


@dataclass
class MarketSession(AggregateRoot):
    """
    Represents a trading session for a specific date.
    
    Aggregates market-wide information for analysis.
    """
    
    id: EntityId
    trading_day: TradingDay
    session_type: str  # 'regular', 'pre_market', 'after_hours'
    start_time: MarketTime
    end_time: MarketTime
    is_active: bool = True
    total_volume: Optional[int] = None
    advancing_issues: Optional[int] = None
    declining_issues: Optional[int] = None
    unchanged_issues: Optional[int] = None
    new_highs: Optional[int] = None
    new_lows: Optional[int] = None
    
    def validate(self) -> None:
        if not self.trading_day:
            raise ValueError("Trading day is required")
        
        if not self.session_type:
            raise ValueError("Session type is required")
        
        valid_sessions = ['regular', 'pre_market', 'after_hours']
        if self.session_type not in valid_sessions:
            raise ValueError(f"Session type must be one of: {valid_sessions}")
        
        if self.start_time and self.end_time:
            if self.start_time.time_value >= self.end_time.time_value:
                raise ValueError("Start time must be before end time")
    
    def update_market_internals(self, 
                               advancing: int,
                               declining: int, 
                               unchanged: int,
                               new_highs: int,
                               new_lows: int) -> None:
        """Update market internals data."""
        self.advancing_issues = advancing
        self.declining_issues = declining
        self.unchanged_issues = unchanged
        self.new_highs = new_highs
        self.new_lows = new_lows
        self.mark_updated()
    
    def get_advance_decline_ratio(self) -> Optional[float]:
        """Calculate advance/decline ratio."""
        if not self.advancing_issues or not self.declining_issues:
            return None
        
        if self.declining_issues == 0:
            return float('inf') if self.advancing_issues > 0 else 0.0
        
        return self.advancing_issues / self.declining_issues
    
    def get_new_high_low_ratio(self) -> Optional[float]:
        """Calculate new highs/new lows ratio."""
        if not self.new_highs or not self.new_lows:
            return None
        
        if self.new_lows == 0:
            return float('inf') if self.new_highs > 0 else 0.0
        
        return self.new_highs / self.new_lows
    
    def get_total_issues(self) -> Optional[int]:
        """Get total number of issues."""
        if all(x is not None for x in [self.advancing_issues, self.declining_issues, self.unchanged_issues]):
            return self.advancing_issues + self.declining_issues + self.unchanged_issues
        return None