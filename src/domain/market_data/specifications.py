"""
Market data specifications for querying and filtering.
"""

from datetime import date
from typing import List

from ..shared.repositories import Specification
from ..shared.identifiers import TickerId
from .entities import Ticker, OHLCV, Quote, MarketSession
from .value_objects import TradingDay


class TickerBySymbolSpecification(Specification[Ticker]):
    """Specification to find ticker by symbol."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
    
    def is_satisfied_by(self, ticker: Ticker) -> bool:
        return ticker.symbol == self.symbol


class ActiveTickersSpecification(Specification[Ticker]):
    """Specification to find active tickers."""
    
    def __init__(self, as_of_date: date = None):
        self.as_of_date = as_of_date or date.today()
    
    def is_satisfied_by(self, ticker: Ticker) -> bool:
        return ticker.is_listed(self.as_of_date)


class TickersBySectorSpecification(Specification[Ticker]):
    """Specification to find tickers by sector."""
    
    def __init__(self, sector: str):
        self.sector = sector
    
    def is_satisfied_by(self, ticker: Ticker) -> bool:
        return ticker.sector == self.sector


class TickersByExchangeSpecification(Specification[Ticker]):
    """Specification to find tickers by exchange."""
    
    def __init__(self, exchange: str):
        self.exchange = exchange
    
    def is_satisfied_by(self, ticker: Ticker) -> bool:
        return ticker.exchange == self.exchange


class OHLCVByTickerSpecification(Specification[OHLCV]):
    """Specification to find OHLCV data by ticker."""
    
    def __init__(self, ticker_id: TickerId):
        self.ticker_id = ticker_id
    
    def is_satisfied_by(self, ohlcv: OHLCV) -> bool:
        return ohlcv.ticker_id == self.ticker_id


class OHLCVByDateSpecification(Specification[OHLCV]):
    """Specification to find OHLCV data by specific date."""
    
    def __init__(self, trading_day: TradingDay):
        self.trading_day = trading_day
    
    def is_satisfied_by(self, ohlcv: OHLCV) -> bool:
        return ohlcv.trading_day.date_value == self.trading_day.date_value


class OHLCVByDateRangeSpecification(Specification[OHLCV]):
    """Specification to find OHLCV data within date range."""
    
    def __init__(self, start_date: date, end_date: date):
        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date")
        self.start_date = start_date
        self.end_date = end_date
    
    def is_satisfied_by(self, ohlcv: OHLCV) -> bool:
        trading_date = ohlcv.trading_day.date_value
        return self.start_date <= trading_date <= self.end_date


class OHLCVWithVolumeAboveSpecification(Specification[OHLCV]):
    """Specification to find OHLCV data with volume above threshold."""
    
    def __init__(self, min_volume: int):
        self.min_volume = min_volume
    
    def is_satisfied_by(self, ohlcv: OHLCV) -> bool:
        return ohlcv.data.volume.value >= self.min_volume


class OHLCVWithPriceAboveSpecification(Specification[OHLCV]):
    """Specification to find OHLCV data with close price above threshold."""
    
    def __init__(self, min_price: float):
        self.min_price = min_price
    
    def is_satisfied_by(self, ohlcv: OHLCV) -> bool:
        return float(ohlcv.data.close_price.value) >= self.min_price


class QuoteByTickerSpecification(Specification[Quote]):
    """Specification to find quotes by ticker."""
    
    def __init__(self, ticker_id: TickerId):
        self.ticker_id = ticker_id
    
    def is_satisfied_by(self, quote: Quote) -> bool:
        return quote.ticker_id == self.ticker_id


class RealtimeQuotesSpecification(Specification[Quote]):
    """Specification to find real-time quotes."""
    
    def is_satisfied_by(self, quote: Quote) -> bool:
        return quote.is_real_time


class ActiveTradingDaySpecification(Specification[MarketSession]):
    """Specification to find active trading sessions."""
    
    def is_satisfied_by(self, session: MarketSession) -> bool:
        return session.is_active and session.trading_day.is_trading_day


class RegularSessionSpecification(Specification[MarketSession]):
    """Specification to find regular trading sessions."""
    
    def is_satisfied_by(self, session: MarketSession) -> bool:
        return session.session_type == 'regular'


# Composite specifications for common queries
class LiquidStocksSpecification(Specification[OHLCV]):
    """Specification for liquid stocks (high volume, reasonable price)."""
    
    def __init__(self, min_volume: int = 100000, min_price: float = 5.0, max_price: float = 1000.0):
        self.volume_spec = OHLCVWithVolumeAboveSpecification(min_volume)
        self.min_price_spec = OHLCVWithPriceAboveSpecification(min_price)
        self.max_price_spec = OHLCVWithPriceBelowSpecification(max_price)
    
    def is_satisfied_by(self, ohlcv: OHLCV) -> bool:
        return (self.volume_spec.is_satisfied_by(ohlcv) and 
                self.min_price_spec.is_satisfied_by(ohlcv) and
                self.max_price_spec.is_satisfied_by(ohlcv))


class OHLCVWithPriceBelowSpecification(Specification[OHLCV]):
    """Specification to find OHLCV data with close price below threshold."""
    
    def __init__(self, max_price: float):
        self.max_price = max_price
    
    def is_satisfied_by(self, ohlcv: OHLCV) -> bool:
        return float(ohlcv.data.close_price.value) <= self.max_price


class TrendingUpSpecification(Specification[OHLCV]):
    """Specification for stocks with upward price trend."""
    
    def is_satisfied_by(self, ohlcv: OHLCV) -> bool:
        return ohlcv.data.is_up_day()


class HighVolumeBreakoutSpecification(Specification[OHLCV]):
    """Specification for high volume breakout patterns."""
    
    def __init__(self, volume_multiplier: float = 2.0):
        self.volume_multiplier = volume_multiplier
    
    def is_satisfied_by(self, ohlcv: OHLCV) -> bool:
        # This would need historical context to compare volume
        # For now, just check if it's an up day with high volume
        return ohlcv.data.is_up_day() and ohlcv.data.volume.value > 0