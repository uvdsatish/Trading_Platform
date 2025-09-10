"""
Technical analysis specifications for querying and filtering.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import List

from ..shared.repositories import Specification
from ..shared.identifiers import TickerId, StrategyId
from .entities import Indicator, Signal, TechnicalPattern, AnalysisResult, Strategy
from .value_objects import (
    IndicatorType, SignalType, SignalStrength, PatternType, 
    TrendDirection, AnalysisPeriod
)


# Indicator Specifications
class IndicatorsByTickerSpecification(Specification[Indicator]):
    """Specification to find indicators by ticker."""
    
    def __init__(self, ticker_id: TickerId):
        self.ticker_id = ticker_id
    
    def is_satisfied_by(self, indicator: Indicator) -> bool:
        return indicator.ticker_id == self.ticker_id


class IndicatorsByTypeSpecification(Specification[Indicator]):
    """Specification to find indicators by type."""
    
    def __init__(self, indicator_type: IndicatorType):
        self.indicator_type = indicator_type
    
    def is_satisfied_by(self, indicator: Indicator) -> bool:
        return indicator.indicator_type == self.indicator_type


class IndicatorsByDateSpecification(Specification[Indicator]):
    """Specification to find indicators by trading date."""
    
    def __init__(self, trading_date: date):
        self.trading_date = trading_date
    
    def is_satisfied_by(self, indicator: Indicator) -> bool:
        return indicator.trading_day.date_value == self.trading_date


class IndicatorsByDateRangeSpecification(Specification[Indicator]):
    """Specification to find indicators within date range."""
    
    def __init__(self, start_date: date, end_date: date):
        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date")
        self.start_date = start_date
        self.end_date = end_date
    
    def is_satisfied_by(self, indicator: Indicator) -> bool:
        trading_date = indicator.trading_day.date_value
        return self.start_date <= trading_date <= self.end_date


class OverboughtIndicatorsSpecification(Specification[Indicator]):
    """Specification to find overbought indicators."""
    
    def __init__(self, threshold: Decimal = None):
        self.threshold = threshold
    
    def is_satisfied_by(self, indicator: Indicator) -> bool:
        return indicator.is_overbought(self.threshold)


class OversoldIndicatorsSpecification(Specification[Indicator]):
    """Specification to find oversold indicators."""
    
    def __init__(self, threshold: Decimal = None):
        self.threshold = threshold
    
    def is_satisfied_by(self, indicator: Indicator) -> bool:
        return indicator.is_oversold(self.threshold)


class IndicatorsByPeriodSpecification(Specification[Indicator]):
    """Specification to find indicators by calculation period."""
    
    def __init__(self, period: int):
        self.period = period
    
    def is_satisfied_by(self, indicator: Indicator) -> bool:
        return indicator.period == self.period


class RSIIndicatorsSpecification(Specification[Indicator]):
    """Specification for RSI indicators with common filters."""
    
    def __init__(self, period: int = 14):
        self.type_spec = IndicatorsByTypeSpecification(IndicatorType.RSI)
        self.period_spec = IndicatorsByPeriodSpecification(period)
    
    def is_satisfied_by(self, indicator: Indicator) -> bool:
        return (self.type_spec.is_satisfied_by(indicator) and 
                self.period_spec.is_satisfied_by(indicator))


# Signal Specifications
class SignalsByTickerSpecification(Specification[Signal]):
    """Specification to find signals by ticker."""
    
    def __init__(self, ticker_id: TickerId):
        self.ticker_id = ticker_id
    
    def is_satisfied_by(self, signal: Signal) -> bool:
        return signal.ticker_id == self.ticker_id


class SignalsByStrategySpecification(Specification[Signal]):
    """Specification to find signals by strategy."""
    
    def __init__(self, strategy_id: StrategyId):
        self.strategy_id = strategy_id
    
    def is_satisfied_by(self, signal: Signal) -> bool:
        return signal.strategy_id == self.strategy_id


class SignalsByTypeSpecification(Specification[Signal]):
    """Specification to find signals by type."""
    
    def __init__(self, signal_type: SignalType):
        self.signal_type = signal_type
    
    def is_satisfied_by(self, signal: Signal) -> bool:
        return signal.signal_type == self.signal_type


class SignalsByStrengthSpecification(Specification[Signal]):
    """Specification to find signals by strength."""
    
    def __init__(self, strength: SignalStrength):
        self.strength = strength
    
    def is_satisfied_by(self, signal: Signal) -> bool:
        return signal.strength == self.strength


class ActiveSignalsSpecification(Specification[Signal]):
    """Specification to find active signals."""
    
    def is_satisfied_by(self, signal: Signal) -> bool:
        return signal.is_active


class SignalsByDateRangeSpecification(Specification[Signal]):
    """Specification to find signals within date range."""
    
    def __init__(self, start_date: datetime, end_date: datetime):
        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date")
        self.start_date = start_date
        self.end_date = end_date
    
    def is_satisfied_by(self, signal: Signal) -> bool:
        return self.start_date <= signal.generated_at <= self.end_date


class BullishSignalsSpecification(Specification[Signal]):
    """Specification to find bullish signals."""
    
    def is_satisfied_by(self, signal: Signal) -> bool:
        return signal.is_bullish()


class BearishSignalsSpecification(Specification[Signal]):
    """Specification to find bearish signals."""
    
    def is_satisfied_by(self, signal: Signal) -> bool:
        return signal.is_bearish()


class StrongSignalsSpecification(Specification[Signal]):
    """Specification to find strong signals."""
    
    def is_satisfied_by(self, signal: Signal) -> bool:
        return signal.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]


class SignalsWithTargetPriceSpecification(Specification[Signal]):
    """Specification to find signals with target prices."""
    
    def is_satisfied_by(self, signal: Signal) -> bool:
        return signal.target_price is not None


class SignalsWithStopLossSpecification(Specification[Signal]):
    """Specification to find signals with stop loss levels."""
    
    def is_satisfied_by(self, signal: Signal) -> bool:
        return signal.stop_loss_price is not None


class HighConfidenceSignalsSpecification(Specification[Signal]):
    """Specification to find high confidence signals."""
    
    def __init__(self, min_confidence: Decimal = Decimal('75')):
        self.min_confidence = min_confidence
    
    def is_satisfied_by(self, signal: Signal) -> bool:
        return (signal.confidence_score is not None and 
                signal.confidence_score >= self.min_confidence)


# Pattern Specifications
class PatternsByTickerSpecification(Specification[TechnicalPattern]):
    """Specification to find patterns by ticker."""
    
    def __init__(self, ticker_id: TickerId):
        self.ticker_id = ticker_id
    
    def is_satisfied_by(self, pattern: TechnicalPattern) -> bool:
        return pattern.ticker_id == self.ticker_id


class PatternsByTypeSpecification(Specification[TechnicalPattern]):
    """Specification to find patterns by type."""
    
    def __init__(self, pattern_type: PatternType):
        self.pattern_type = pattern_type
    
    def is_satisfied_by(self, pattern: TechnicalPattern) -> bool:
        return pattern.pattern_type == self.pattern_type


class PatternsByDateRangeSpecification(Specification[TechnicalPattern]):
    """Specification to find patterns within date range."""
    
    def __init__(self, start_date: date, end_date: date):
        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date")
        self.start_date = start_date
        self.end_date = end_date
    
    def is_satisfied_by(self, pattern: TechnicalPattern) -> bool:
        return (self.start_date <= pattern.start_date <= self.end_date or
                self.start_date <= pattern.end_date <= self.end_date)


class BullishPatternsSpecification(Specification[TechnicalPattern]):
    """Specification to find bullish patterns."""
    
    def is_satisfied_by(self, pattern: TechnicalPattern) -> bool:
        return pattern.is_bullish_pattern()


class BearishPatternsSpecification(Specification[TechnicalPattern]):
    """Specification to find bearish patterns."""
    
    def is_satisfied_by(self, pattern: TechnicalPattern) -> bool:
        return pattern.is_bearish_pattern()


class BreakoutPatternsSpecification(Specification[TechnicalPattern]):
    """Specification to find breakout patterns."""
    
    def is_satisfied_by(self, pattern: TechnicalPattern) -> bool:
        return pattern.is_breakout_pattern()


class HighConfidencePatternsSpecification(Specification[TechnicalPattern]):
    """Specification to find high confidence patterns."""
    
    def __init__(self, min_confidence: Decimal = Decimal('70')):
        self.min_confidence = min_confidence
    
    def is_satisfied_by(self, pattern: TechnicalPattern) -> bool:
        return pattern.confidence >= self.min_confidence


class VolumeConfirmedPatternsSpecification(Specification[TechnicalPattern]):
    """Specification to find volume-confirmed patterns."""
    
    def is_satisfied_by(self, pattern: TechnicalPattern) -> bool:
        return pattern.volume_confirmation


# Strategy Specifications
class ActiveStrategiesSpecification(Specification[Strategy]):
    """Specification to find active strategies."""
    
    def is_satisfied_by(self, strategy: Strategy) -> bool:
        return strategy.is_active


class StrategiesByTypeSpecification(Specification[Strategy]):
    """Specification to find strategies by type."""
    
    def __init__(self, strategy_type: str):
        self.strategy_type = strategy_type
    
    def is_satisfied_by(self, strategy: Strategy) -> bool:
        return strategy.strategy_type == self.strategy_type


class RecentlyRunStrategiesSpecification(Specification[Strategy]):
    """Specification to find recently run strategies."""
    
    def __init__(self, hours: int = 24):
        self.hours = hours
    
    def is_satisfied_by(self, strategy: Strategy) -> bool:
        return strategy.has_run_recently(self.hours)


# Analysis Result Specifications
class AnalysisResultsByTickerSpecification(Specification[AnalysisResult]):
    """Specification to find analysis results by ticker."""
    
    def __init__(self, ticker_id: TickerId):
        self.ticker_id = ticker_id
    
    def is_satisfied_by(self, result: AnalysisResult) -> bool:
        return result.ticker_id == self.ticker_id


class AnalysisResultsByDateSpecification(Specification[AnalysisResult]):
    """Specification to find analysis results by date."""
    
    def __init__(self, analysis_date: date):
        self.analysis_date = analysis_date
    
    def is_satisfied_by(self, result: AnalysisResult) -> bool:
        return result.analysis_date == self.analysis_date


class BullishAnalysisResultsSpecification(Specification[AnalysisResult]):
    """Specification to find bullish analysis results."""
    
    def is_satisfied_by(self, result: AnalysisResult) -> bool:
        return result.overall_trend == TrendDirection.BULLISH


class BearishAnalysisResultsSpecification(Specification[AnalysisResult]):
    """Specification to find bearish analysis results."""
    
    def is_satisfied_by(self, result: AnalysisResult) -> bool:
        return result.overall_trend == TrendDirection.BEARISH


class HighScoreAnalysisResultsSpecification(Specification[AnalysisResult]):
    """Specification to find high composite score analysis results."""
    
    def __init__(self, min_score: Decimal = Decimal('50')):
        self.min_score = min_score
    
    def is_satisfied_by(self, result: AnalysisResult) -> bool:
        return (result.composite_score is not None and 
                result.composite_score >= self.min_score)


class BuyRecommendationSpecification(Specification[AnalysisResult]):
    """Specification to find buy recommendations."""
    
    def is_satisfied_by(self, result: AnalysisResult) -> bool:
        return result.recommendation in [SignalType.BUY, SignalType.STRONG_BUY]


class SellRecommendationSpecification(Specification[AnalysisResult]):
    """Specification to find sell recommendations."""
    
    def is_satisfied_by(self, result: AnalysisResult) -> bool:
        return result.recommendation in [SignalType.SELL, SignalType.STRONG_SELL]


# Composite Specifications for Common Trading Queries
class RunAwayMomentumSpecification(Specification[Signal]):
    """Specification for RunAway momentum signals."""
    
    def __init__(self, strategy_name: str = "RunAway"):
        self.active_spec = ActiveSignalsSpecification()
        self.bullish_spec = BullishSignalsSpecification()
        self.strong_spec = StrongSignalsSpecification()
        self.strategy_name = strategy_name
    
    def is_satisfied_by(self, signal: Signal) -> bool:
        strategy_match = str(signal.strategy_id.value).lower() == self.strategy_name.lower()
        return (strategy_match and
                self.active_spec.is_satisfied_by(signal) and
                self.bullish_spec.is_satisfied_by(signal) and
                self.strong_spec.is_satisfied_by(signal))


class BreakoutCandidatesSpecification(Specification[TechnicalPattern]):
    """Specification for breakout candidates."""
    
    def __init__(self, min_confidence: Decimal = Decimal('70')):
        self.breakout_spec = BreakoutPatternsSpecification()
        self.confidence_spec = HighConfidencePatternsSpecification(min_confidence)
        self.volume_spec = VolumeConfirmedPatternsSpecification()
    
    def is_satisfied_by(self, pattern: TechnicalPattern) -> bool:
        return (self.breakout_spec.is_satisfied_by(pattern) and
                self.confidence_spec.is_satisfied_by(pattern) and
                self.volume_spec.is_satisfied_by(pattern))


class OversoldBounceSpecification(Specification[Indicator]):
    """Specification for oversold bounce candidates."""
    
    def __init__(self, rsi_threshold: Decimal = Decimal('30')):
        self.rsi_spec = RSIIndicatorsSpecification()
        self.oversold_spec = OversoldIndicatorsSpecification(rsi_threshold)
    
    def is_satisfied_by(self, indicator: Indicator) -> bool:
        return (self.rsi_spec.is_satisfied_by(indicator) and
                self.oversold_spec.is_satisfied_by(indicator))


class HighQualitySignalsSpecification(Specification[Signal]):
    """Specification for high quality trading signals."""
    
    def __init__(self, min_confidence: Decimal = Decimal('80')):
        self.active_spec = ActiveSignalsSpecification()
        self.confidence_spec = HighConfidenceSignalsSpecification(min_confidence)
        self.target_spec = SignalsWithTargetPriceSpecification()
        self.stop_spec = SignalsWithStopLossSpecification()
    
    def is_satisfied_by(self, signal: Signal) -> bool:
        return (self.active_spec.is_satisfied_by(signal) and
                self.confidence_spec.is_satisfied_by(signal) and
                self.target_spec.is_satisfied_by(signal) and
                self.stop_spec.is_satisfied_by(signal))


class TechnicalScreenerSpecification(Specification[AnalysisResult]):
    """Specification for technical analysis screening."""
    
    def __init__(self, 
                 trend: TrendDirection = TrendDirection.BULLISH,
                 min_score: Decimal = Decimal('60'),
                 min_confidence: Decimal = Decimal('70')):
        self.trend = trend
        self.min_score = min_score
        self.min_confidence = min_confidence
    
    def is_satisfied_by(self, result: AnalysisResult) -> bool:
        trend_match = result.overall_trend == self.trend
        
        score_match = (result.composite_score is not None and 
                      result.composite_score >= self.min_score)
        
        confidence_match = (result.confidence_level is not None and
                           result.confidence_level >= self.min_confidence)
        
        return trend_match and score_match and confidence_match