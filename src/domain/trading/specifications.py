"""
Trading specifications for querying and filtering.
"""

from datetime import date, datetime
from typing import List

from ..shared.repositories import Specification
from ..shared.identifiers import TickerId, PortfolioId, OrderId
from .entities import Order, Position, Portfolio, Trade, Account
from .value_objects import OrderStatus, OrderSide, PositionSide, OrderType


class OrdersByStatusSpecification(Specification[Order]):
    """Specification to find orders by status."""
    
    def __init__(self, status: OrderStatus):
        self.status = status
    
    def is_satisfied_by(self, order: Order) -> bool:
        return order.status == self.status


class OrdersByTickerSpecification(Specification[Order]):
    """Specification to find orders by ticker."""
    
    def __init__(self, ticker_id: TickerId):
        self.ticker_id = ticker_id
    
    def is_satisfied_by(self, order: Order) -> bool:
        return order.ticker_id == self.ticker_id


class OrdersByPortfolioSpecification(Specification[Order]):
    """Specification to find orders by portfolio."""
    
    def __init__(self, portfolio_id: PortfolioId):
        self.portfolio_id = portfolio_id
    
    def is_satisfied_by(self, order: Order) -> bool:
        return order.portfolio_id == self.portfolio_id


class OrdersBySideSpecification(Specification[Order]):
    """Specification to find orders by side (buy/sell)."""
    
    def __init__(self, side: OrderSide):
        self.side = side
    
    def is_satisfied_by(self, order: Order) -> bool:
        return order.side == self.side


class OrdersByTypeSpecification(Specification[Order]):
    """Specification to find orders by type."""
    
    def __init__(self, order_type: OrderType):
        self.order_type = order_type
    
    def is_satisfied_by(self, order: Order) -> bool:
        return order.order_type == self.order_type


class ActiveOrdersSpecification(Specification[Order]):
    """Specification to find active orders."""
    
    def is_satisfied_by(self, order: Order) -> bool:
        return order.is_active()


class FilledOrdersSpecification(Specification[Order]):
    """Specification to find filled orders."""
    
    def is_satisfied_by(self, order: Order) -> bool:
        return order.is_filled()


class OrdersByDateRangeSpecification(Specification[Order]):
    """Specification to find orders within date range."""
    
    def __init__(self, start_date: datetime, end_date: datetime):
        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date")
        self.start_date = start_date
        self.end_date = end_date
    
    def is_satisfied_by(self, order: Order) -> bool:
        if not order.submitted_at:
            return False
        return self.start_date <= order.submitted_at <= self.end_date


class PositionsByPortfolioSpecification(Specification[Position]):
    """Specification to find positions by portfolio."""
    
    def __init__(self, portfolio_id: PortfolioId):
        self.portfolio_id = portfolio_id
    
    def is_satisfied_by(self, position: Position) -> bool:
        return position.portfolio_id == self.portfolio_id


class PositionsByTickerSpecification(Specification[Position]):
    """Specification to find positions by ticker."""
    
    def __init__(self, ticker_id: TickerId):
        self.ticker_id = ticker_id
    
    def is_satisfied_by(self, position: Position) -> bool:
        return position.ticker_id == self.ticker_id


class PositionsBySideSpecification(Specification[Position]):
    """Specification to find positions by side."""
    
    def __init__(self, side: PositionSide):
        self.side = side
    
    def is_satisfied_by(self, position: Position) -> bool:
        return position.side == self.side


class OpenPositionsSpecification(Specification[Position]):
    """Specification to find open (non-flat) positions."""
    
    def is_satisfied_by(self, position: Position) -> bool:
        return not position.is_flat()


class LongPositionsSpecification(Specification[Position]):
    """Specification to find long positions."""
    
    def is_satisfied_by(self, position: Position) -> bool:
        return position.is_long()


class ShortPositionsSpecification(Specification[Position]):
    """Specification to find short positions."""
    
    def is_satisfied_by(self, position: Position) -> bool:
        return position.is_short()


class ProfitablePositionsSpecification(Specification[Position]):
    """Specification to find profitable positions."""
    
    def is_satisfied_by(self, position: Position) -> bool:
        pnl = position.get_unrealized_pnl()
        return pnl is not None and pnl.amount > 0


class LosingPositionsSpecification(Specification[Position]):
    """Specification to find losing positions."""
    
    def is_satisfied_by(self, position: Position) -> bool:
        pnl = position.get_unrealized_pnl()
        return pnl is not None and pnl.amount < 0


class PositionsAboveValueSpecification(Specification[Position]):
    """Specification to find positions above certain market value."""
    
    def __init__(self, min_value: float):
        self.min_value = min_value
    
    def is_satisfied_by(self, position: Position) -> bool:
        market_value = position.get_market_value()
        return market_value is not None and float(market_value.amount) >= self.min_value


class TradesByDateRangeSpecification(Specification[Trade]):
    """Specification to find trades within date range."""
    
    def __init__(self, start_date: datetime, end_date: datetime):
        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date")
        self.start_date = start_date
        self.end_date = end_date
    
    def is_satisfied_by(self, trade: Trade) -> bool:
        return self.start_date <= trade.execution.execution_time <= self.end_date


class TradesByTickerSpecification(Specification[Trade]):
    """Specification to find trades by ticker."""
    
    def __init__(self, ticker_id: TickerId):
        self.ticker_id = ticker_id
    
    def is_satisfied_by(self, trade: Trade) -> bool:
        return trade.ticker_id == self.ticker_id


class TradesBySideSpecification(Specification[Trade]):
    """Specification to find trades by side."""
    
    def __init__(self, side: OrderSide):
        self.side = side
    
    def is_satisfied_by(self, trade: Trade) -> bool:
        return trade.side == self.side


class TradesByStrategySpecification(Specification[Trade]):
    """Specification to find trades by strategy."""
    
    def __init__(self, strategy_id: str):
        self.strategy_id = strategy_id
    
    def is_satisfied_by(self, trade: Trade) -> bool:
        return trade.strategy_id == self.strategy_id


class TradesAboveAmountSpecification(Specification[Trade]):
    """Specification to find trades above certain amount."""
    
    def __init__(self, min_amount: float):
        self.min_amount = min_amount
    
    def is_satisfied_by(self, trade: Trade) -> bool:
        gross_amount = trade.get_gross_amount()
        return float(gross_amount.amount) >= self.min_amount


class PortfoliosByAccountSpecification(Specification[Portfolio]):
    """Specification to find portfolios by account."""
    
    def __init__(self, account_id: str):
        self.account_id = account_id
    
    def is_satisfied_by(self, portfolio: Portfolio) -> bool:
        return str(portfolio.account_id.value) == self.account_id


class PortfoliosAboveValueSpecification(Specification[Portfolio]):
    """Specification to find portfolios above certain value."""
    
    def __init__(self, min_value: float):
        self.min_value = min_value
    
    def is_satisfied_by(self, portfolio: Portfolio) -> bool:
        total_value = portfolio.get_total_portfolio_value()
        return float(total_value.amount) >= self.min_value


class ActiveAccountsSpecification(Specification[Account]):
    """Specification to find active accounts."""
    
    def is_satisfied_by(self, account: Account) -> bool:
        return account.is_active


class AccountsByTypeSpecification(Specification[Account]):
    """Specification to find accounts by type."""
    
    def __init__(self, account_type: str):
        self.account_type = account_type
    
    def is_satisfied_by(self, account: Account) -> bool:
        return account.account_type == self.account_type


# Composite specifications for common trading queries
class DayTradingOrdersSpecification(Specification[Order]):
    """Specification for day trading orders (buy and sell same day)."""
    
    def __init__(self, trading_date: date):
        self.trading_date = trading_date
    
    def is_satisfied_by(self, order: Order) -> bool:
        if not order.submitted_at:
            return False
        return order.submitted_at.date() == self.trading_date


class LargePositionsSpecification(Specification[Position]):
    """Specification for large positions (above certain threshold)."""
    
    def __init__(self, min_value: float = 10000.0):
        self.value_spec = PositionsAboveValueSpecification(min_value)
        self.open_spec = OpenPositionsSpecification()
    
    def is_satisfied_by(self, position: Position) -> bool:
        return (self.value_spec.is_satisfied_by(position) and 
                self.open_spec.is_satisfied_by(position))


class HighRiskPositionsSpecification(Specification[Position]):
    """Specification for high risk positions (losing positions above threshold)."""
    
    def __init__(self, max_loss_percent: float = -10.0):
        self.max_loss_percent = max_loss_percent
    
    def is_satisfied_by(self, position: Position) -> bool:
        pnl_percent = position.get_unrealized_pnl_percent()
        return (pnl_percent is not None and 
                pnl_percent <= self.max_loss_percent and
                not position.is_flat())