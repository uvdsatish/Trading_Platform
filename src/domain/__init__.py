"""
Domain layer for the trading platform.

This module contains the core business logic and domain models following
Domain-Driven Design (DDD) principles. The domain layer is independent
of infrastructure concerns and focuses on business rules and entities.

Domain Structure:
- shared/: Common domain concepts (base entities, value objects, events)
- market_data/: Market data entities (OHLCV, quotes, tickers)
- trading/: Trading domain (positions, orders, portfolios)
- technical_analysis/: Technical indicators and signals
- backtesting/: Backtesting models and results

Usage:
    from src.domain.market_data import OHLCV, Ticker
    from src.domain.trading import Position, Order
    from src.domain.technical_analysis import Indicator, Signal
"""

from .shared import *

__all__ = [
    # Re-export shared domain concepts
    'Entity',
    'ValueObject', 
    'AggregateRoot',
    'DomainEvent',
    'DomainService',
    'Repository',
    'Specification',
]