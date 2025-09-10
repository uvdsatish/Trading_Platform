"""
Technical analysis value objects.
"""

from dataclasses import dataclass
from enum import Enum
from decimal import Decimal
from typing import Optional, List, Dict, Any
from datetime import date

from ..shared.value_objects import ValueObject, Percentage


class IndicatorType(Enum):
    """Types of technical indicators."""
    # Trend indicators
    SMA = "SMA"                    # Simple Moving Average
    EMA = "EMA"                    # Exponential Moving Average
    WMA = "WMA"                    # Weighted Moving Average
    MACD = "MACD"                  # Moving Average Convergence Divergence
    ADX = "ADX"                    # Average Directional Index
    
    # Momentum indicators
    RSI = "RSI"                    # Relative Strength Index
    STOCH = "STOCH"                # Stochastic Oscillator
    ROC = "ROC"                    # Rate of Change
    MOM = "MOM"                    # Momentum
    CCI = "CCI"                    # Commodity Channel Index
    
    # Volume indicators
    OBV = "OBV"                    # On-Balance Volume
    VOLUME_SMA = "VOLUME_SMA"      # Volume Moving Average
    VWAP = "VWAP"                  # Volume Weighted Average Price
    
    # Volatility indicators
    BB = "BB"                      # Bollinger Bands
    ATR = "ATR"                    # Average True Range
    KELT = "KELT"                  # Keltner Channels
    
    # Support/Resistance
    PIVOT = "PIVOT"                # Pivot Points
    FIBR = "FIBR"                  # Fibonacci Retracements
    
    # Custom indicators (your proprietary ones)
    PLURALITY_RS = "PLURALITY_RS"  # Plurality Relative Strength
    RUNAWAY = "RUNAWAY"            # RunAway momentum
    BREAKOUT_52W = "BREAKOUT_52W"  # 52-week breakout
    HINDENBURG = "HINDENBURG"      # Hindenburg Omen


class SignalType(Enum):
    """Types of trading signals."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"


class TrendDirection(Enum):
    """Trend direction."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


class PatternType(Enum):
    """Chart pattern types."""
    # Reversal patterns
    HEAD_AND_SHOULDERS = "HEAD_AND_SHOULDERS"
    DOUBLE_TOP = "DOUBLE_TOP"
    DOUBLE_BOTTOM = "DOUBLE_BOTTOM"
    TRIPLE_TOP = "TRIPLE_TOP"
    TRIPLE_BOTTOM = "TRIPLE_BOTTOM"
    
    # Continuation patterns
    FLAG = "FLAG"
    PENNANT = "PENNANT"
    TRIANGLE = "TRIANGLE"
    WEDGE = "WEDGE"
    RECTANGLE = "RECTANGLE"
    
    # Candlestick patterns
    DOJI = "DOJI"
    HAMMER = "HAMMER"
    SHOOTING_STAR = "SHOOTING_STAR"
    ENGULFING = "ENGULFING"
    
    # Breakout patterns
    BREAKOUT_52W_HIGH = "BREAKOUT_52W_HIGH"
    BREAKOUT_25D_HIGH = "BREAKOUT_25D_HIGH"
    BREAKOUT_100D_HIGH = "BREAKOUT_100D_HIGH"


class MovingAverageType(Enum):
    """Moving average calculation types."""
    SIMPLE = "SIMPLE"
    EXPONENTIAL = "EXPONENTIAL"
    WEIGHTED = "WEIGHTED"
    VOLUME_WEIGHTED = "VOLUME_WEIGHTED"


@dataclass(frozen=True)
class IndicatorValue(ValueObject):
    """Represents a calculated indicator value."""
    
    value: Decimal
    normalized_value: Optional[Decimal] = None  # 0-100 scale for oscillators
    signal_line: Optional[Decimal] = None       # For indicators with signal lines
    histogram: Optional[Decimal] = None         # For MACD histogram
    
    def validate(self) -> None:
        if self.value is None:
            raise ValueError("Indicator value cannot be None")
        
        if self.normalized_value is not None:
            if not (0 <= self.normalized_value <= 100):
                raise ValueError("Normalized value must be between 0 and 100")
    
    def is_overbought(self, threshold: Decimal = Decimal('70')) -> bool:
        """Check if indicator suggests overbought condition."""
        if self.normalized_value is None:
            return False
        return self.normalized_value >= threshold
    
    def is_oversold(self, threshold: Decimal = Decimal('30')) -> bool:
        """Check if indicator suggests oversold condition."""
        if self.normalized_value is None:
            return False
        return self.normalized_value <= threshold
    
    def is_bullish_crossover(self) -> bool:
        """Check if there's a bullish crossover (value > signal line)."""
        if self.signal_line is None:
            return False
        return self.value > self.signal_line
    
    def is_bearish_crossover(self) -> bool:
        """Check if there's a bearish crossover (value < signal line)."""
        if self.signal_line is None:
            return False
        return self.value < self.signal_line


@dataclass(frozen=True)
class SignalCondition(ValueObject):
    """Represents a condition that generates a signal."""
    
    indicator_type: IndicatorType
    condition: str  # e.g., "RSI > 70", "MACD crossover", "breakout"
    threshold: Optional[Decimal] = None
    lookback_periods: int = 1
    
    def validate(self) -> None:
        if not self.condition or not self.condition.strip():
            raise ValueError("Signal condition cannot be empty")
        
        if self.lookback_periods <= 0:
            raise ValueError("Lookback periods must be positive")
    
    def evaluate(self, indicator_value: IndicatorValue) -> bool:
        """Evaluate if the condition is met."""
        # This would contain the logic to evaluate conditions
        # For now, simplified implementation
        if "overbought" in self.condition.lower():
            return indicator_value.is_overbought(self.threshold or Decimal('70'))
        elif "oversold" in self.condition.lower():
            return indicator_value.is_oversold(self.threshold or Decimal('30'))
        elif "crossover" in self.condition.lower():
            return indicator_value.is_bullish_crossover()
        
        return False


@dataclass(frozen=True)
class AnalysisPeriod(ValueObject):
    """Represents a time period for analysis."""
    
    start_date: date
    end_date: date
    period_type: str  # 'daily', 'weekly', 'monthly', 'intraday'
    interval: Optional[str] = None  # '1min', '5min', '1hour' for intraday
    
    def validate(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("Start date must be before or equal to end date")
        
        valid_periods = ['daily', 'weekly', 'monthly', 'intraday']
        if self.period_type not in valid_periods:
            raise ValueError(f"Period type must be one of: {valid_periods}")
        
        if self.period_type == 'intraday' and not self.interval:
            raise ValueError("Interval is required for intraday periods")
    
    def get_duration_days(self) -> int:
        """Get duration in days."""
        return (self.end_date - self.start_date).days
    
    def contains_date(self, check_date: date) -> bool:
        """Check if date falls within this period."""
        return self.start_date <= check_date <= self.end_date


@dataclass(frozen=True)
class BollingerBands(ValueObject):
    """Bollinger Bands indicator values."""
    
    upper_band: Decimal
    middle_band: Decimal  # Usually SMA
    lower_band: Decimal
    bandwidth: Decimal
    percent_b: Decimal    # Position within bands
    
    def validate(self) -> None:
        if not (self.lower_band <= self.middle_band <= self.upper_band):
            raise ValueError("Band order must be: lower <= middle <= upper")
        
        if self.bandwidth < 0:
            raise ValueError("Bandwidth cannot be negative")
    
    def is_squeeze(self, threshold: Decimal = Decimal('0.1')) -> bool:
        """Check if bands are in a squeeze (low volatility)."""
        return self.bandwidth < threshold
    
    def is_expansion(self, previous_bandwidth: Decimal) -> bool:
        """Check if bands are expanding (increasing volatility)."""
        return self.bandwidth > previous_bandwidth
    
    def get_position_in_bands(self, price: Decimal) -> Decimal:
        """Get price position within bands (0-1 scale)."""
        if self.upper_band == self.lower_band:
            return Decimal('0.5')  # Middle when no range
        
        return (price - self.lower_band) / (self.upper_band - self.lower_band)


@dataclass(frozen=True)
class MACDValues(ValueObject):
    """MACD indicator values."""
    
    macd_line: Decimal
    signal_line: Decimal
    histogram: Decimal
    
    def validate(self) -> None:
        # Histogram should equal MACD line minus signal line
        expected_histogram = self.macd_line - self.signal_line
        if abs(self.histogram - expected_histogram) > Decimal('0.001'):
            raise ValueError("Histogram must equal MACD line minus signal line")
    
    def is_bullish_crossover(self) -> bool:
        """Check for bullish MACD crossover."""
        return self.macd_line > self.signal_line and self.histogram > 0
    
    def is_bearish_crossover(self) -> bool:
        """Check for bearish MACD crossover."""
        return self.macd_line < self.signal_line and self.histogram < 0
    
    def is_diverging(self) -> bool:
        """Check if MACD and signal are diverging."""
        return abs(self.histogram) > abs(self.macd_line - self.signal_line) * Decimal('0.5')


@dataclass(frozen=True)
class RelativeStrengthValue(ValueObject):
    """Relative strength indicator value (your Plurality-WAMRS system)."""
    
    rs_rating: Decimal          # 1-100 relative strength rating
    sector_rs: Decimal          # Sector relative strength
    industry_rs: Decimal        # Industry relative strength
    market_rs: Decimal          # Market relative strength
    momentum_score: Decimal     # Momentum component
    
    def validate(self) -> None:
        values = [self.rs_rating, self.sector_rs, self.industry_rs, self.market_rs]
        for value in values:
            if not (0 <= value <= 100):
                raise ValueError("Relative strength values must be between 0 and 100")
    
    def is_strong_performer(self, threshold: Decimal = Decimal('80')) -> bool:
        """Check if stock is a strong relative performer."""
        return self.rs_rating >= threshold
    
    def is_sector_leader(self, threshold: Decimal = Decimal('70')) -> bool:
        """Check if stock is leading its sector."""
        return self.sector_rs >= threshold
    
    def get_composite_score(self) -> Decimal:
        """Calculate composite relative strength score."""
        weights = {
            'rs_rating': Decimal('0.4'),
            'sector_rs': Decimal('0.3'),
            'industry_rs': Decimal('0.2'),
            'momentum_score': Decimal('0.1')
        }
        
        composite = (
            self.rs_rating * weights['rs_rating'] +
            self.sector_rs * weights['sector_rs'] +
            self.industry_rs * weights['industry_rs'] +
            self.momentum_score * weights['momentum_score']
        )
        
        return composite


@dataclass(frozen=True)
class BreakoutMetrics(ValueObject):
    """Breakout analysis metrics."""
    
    breakout_level: Decimal
    volume_ratio: Decimal       # Current volume vs average
    price_change_percent: Decimal
    resistance_strength: int   # Number of times level was tested
    time_since_breakout: int   # Days since breakout
    
    def validate(self) -> None:
        if self.volume_ratio < 0:
            raise ValueError("Volume ratio cannot be negative")
        
        if self.resistance_strength < 0:
            raise ValueError("Resistance strength cannot be negative")
        
        if self.time_since_breakout < 0:
            raise ValueError("Time since breakout cannot be negative")
    
    def is_volume_confirmed(self, min_ratio: Decimal = Decimal('1.5')) -> bool:
        """Check if breakout is confirmed by volume."""
        return self.volume_ratio >= min_ratio
    
    def is_significant_breakout(self, min_change: Decimal = Decimal('2.0')) -> bool:
        """Check if breakout is significant."""
        return abs(self.price_change_percent) >= min_change
    
    def get_breakout_quality_score(self) -> Decimal:
        """Calculate overall breakout quality score (0-100)."""
        volume_score = min(self.volume_ratio * 20, 40)  # Max 40 points
        price_score = min(abs(self.price_change_percent) * 10, 30)  # Max 30 points
        resistance_score = min(self.resistance_strength * 5, 20)  # Max 20 points
        time_score = max(10 - self.time_since_breakout, 0)  # Max 10 points, decreases with time
        
        return volume_score + price_score + resistance_score + time_score