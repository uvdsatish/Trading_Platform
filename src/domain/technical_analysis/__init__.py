"""
Technical analysis domain models.

This module contains entities and value objects related to technical analysis,
including indicators, signals, patterns, and analytical calculations.
"""

from .entities import Indicator, Signal, TechnicalPattern, AnalysisResult, Strategy
from .value_objects import (
    IndicatorType,
    SignalStrength,
    SignalType,
    PatternType,
    TrendDirection,
    IndicatorValue,
    SignalCondition,
    AnalysisPeriod,
    MovingAverageType
)
from .specifications import (
    IndicatorsByTickerSpecification,
    IndicatorsByTypeSpecification,
    SignalsByStrengthSpecification,
    SignalsByDateRangeSpecification,
    BullishSignalsSpecification,
    BearishSignalsSpecification,
    PatternsByTypeSpecification
)

__all__ = [
    # Entities
    'Indicator',
    'Signal',
    'TechnicalPattern',
    'AnalysisResult',
    'Strategy',
    
    # Value Objects
    'IndicatorType',
    'SignalStrength',
    'SignalType',
    'PatternType',
    'TrendDirection',
    'IndicatorValue',
    'SignalCondition',
    'AnalysisPeriod',
    'MovingAverageType',
    
    # Specifications
    'IndicatorsByTickerSpecification',
    'IndicatorsByTypeSpecification',
    'SignalsByStrengthSpecification',
    'SignalsByDateRangeSpecification',
    'BullishSignalsSpecification',
    'BearishSignalsSpecification',
    'PatternsByTypeSpecification',
]