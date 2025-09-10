"""
Trading entities.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from decimal import Decimal

from ..shared.entities import Entity, AggregateRoot
from ..shared.identifiers import OrderId, PositionId, PortfolioId, TickerId, EntityId
from ..shared.events import OrderPlaced, OrderFilled
from ..shared.value_objects import Money, Price, Quantity
from .value_objects import (
    OrderType, OrderSide, OrderStatus, TimeInForce, PositionSide,
    TradeExecution, RiskParameters, PortfolioMetrics, OrderConstraints, PositionMetrics
)


@dataclass
class Order(AggregateRoot):
    """
    Represents a trading order.
    
    An order is an instruction to buy or sell a financial instrument.
    """
    
    id: OrderId
    ticker_id: TickerId
    portfolio_id: PortfolioId
    order_type: OrderType
    side: OrderSide
    quantity: Quantity
    price: Optional[Price] = None
    stop_price: Optional[Price] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: Optional[datetime] = None
    filled_quantity: Quantity = field(default_factory=lambda: Quantity(0))
    average_fill_price: Optional[Price] = None
    executions: List[TradeExecution] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    
    def validate(self) -> None:
        if not self.ticker_id:
            raise ValueError("Ticker ID is required")
        
        if not self.portfolio_id:
            raise ValueError("Portfolio ID is required")
        
        if self.quantity.value <= 0:
            raise ValueError("Order quantity must be positive")
        
        # Validate price requirements based on order type
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if not self.price:
                raise ValueError(f"{self.order_type.value} orders require a price")
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP]:
            if not self.stop_price:
                raise ValueError(f"{self.order_type.value} orders require a stop price")
        
        # Validate filled quantity
        if self.filled_quantity.value > self.quantity.value:
            raise ValueError("Filled quantity cannot exceed order quantity")
    
    def submit(self, submission_time: datetime = None) -> None:
        """Submit the order for execution."""
        if self.status != OrderStatus.PENDING:
            raise ValueError(f"Cannot submit order in {self.status.value} status")
        
        self.status = OrderStatus.SUBMITTED
        self.submitted_at = submission_time or datetime.utcnow()
        self.mark_updated()
        
        # Raise domain event
        event = OrderPlaced(
            aggregate_id=str(self.id.value),
            order_id=str(self.id.value),
            ticker=str(self.ticker_id.value),
            side=self.side.value,
            quantity=self.quantity.value,
            price=self.price
        )
        self.add_domain_event(event)
    
    def add_execution(self, execution: TradeExecution) -> None:
        """Add a trade execution to this order."""
        if self.status not in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            raise ValueError(f"Cannot execute order in {self.status.value} status")
        
        # Validate execution quantity
        remaining_quantity = self.quantity.value - self.filled_quantity.value
        if execution.execution_quantity.value > remaining_quantity:
            raise ValueError("Execution quantity exceeds remaining order quantity")
        
        self.executions.append(execution)
        self.filled_quantity = Quantity(
            self.filled_quantity.value + execution.execution_quantity.value
        )
        
        # Update average fill price
        self._update_average_fill_price()
        
        # Update status
        if self.filled_quantity.value == self.quantity.value:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.mark_updated()
        
        # Raise domain event
        event = OrderFilled(
            aggregate_id=str(self.id.value),
            order_id=str(self.id.value),
            fill_price=execution.execution_price,
            fill_quantity=execution.execution_quantity.value
        )
        self.add_domain_event(event)
    
    def cancel(self, reason: str = None) -> None:
        """Cancel the order."""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise ValueError(f"Cannot cancel order in {self.status.value} status")
        
        self.status = OrderStatus.CANCELLED
        if reason:
            self.rejection_reason = reason
        self.mark_updated()
    
    def reject(self, reason: str) -> None:
        """Reject the order."""
        if self.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
            raise ValueError(f"Cannot reject order in {self.status.value} status")
        
        self.status = OrderStatus.REJECTED
        self.rejection_reason = reason
        self.mark_updated()
    
    def get_remaining_quantity(self) -> Quantity:
        """Get the remaining quantity to be filled."""
        return Quantity(self.quantity.value - self.filled_quantity.value)
    
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    def is_active(self) -> bool:
        """Check if order is active (can still be executed)."""
        return self.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
    
    def _update_average_fill_price(self) -> None:
        """Update the average fill price based on executions."""
        if not self.executions:
            return
        
        total_value = Decimal('0')
        total_quantity = 0
        
        for execution in self.executions:
            total_value += execution.execution_price.value * execution.execution_quantity.value
            total_quantity += execution.execution_quantity.value
        
        if total_quantity > 0:
            self.average_fill_price = Price(total_value / total_quantity)


@dataclass
class Position(AggregateRoot):
    """
    Represents a trading position in a specific instrument.
    
    A position tracks the current holdings in a ticker within a portfolio.
    """
    
    id: PositionId
    portfolio_id: PortfolioId
    ticker_id: TickerId
    side: PositionSide
    quantity: Quantity
    average_price: Price
    current_price: Optional[Price] = None
    cost_basis: Optional[Money] = None
    opened_at: datetime = field(default_factory=datetime.utcnow)
    trades: List['Trade'] = field(default_factory=list)
    
    def validate(self) -> None:
        if not self.portfolio_id:
            raise ValueError("Portfolio ID is required")
        
        if not self.ticker_id:
            raise ValueError("Ticker ID is required")
        
        if self.side == PositionSide.FLAT and self.quantity.value != 0:
            raise ValueError("Flat positions must have zero quantity")
        
        if self.side != PositionSide.FLAT and self.quantity.value == 0:
            raise ValueError("Non-flat positions must have non-zero quantity")
        
        if self.quantity.value < 0:
            raise ValueError("Position quantity cannot be negative")
    
    def add_trade(self, trade: 'Trade') -> None:
        """Add a trade to this position and update metrics."""
        if trade.ticker_id != self.ticker_id:
            raise ValueError("Trade ticker must match position ticker")
        
        self.trades.append(trade)
        self._update_position_metrics()
        self.mark_updated()
    
    def update_market_price(self, new_price: Price) -> None:
        """Update the current market price for P&L calculation."""
        self.current_price = new_price
        self.mark_updated()
    
    def get_market_value(self) -> Optional[Money]:
        """Calculate current market value of the position."""
        if not self.current_price:
            return None
        
        market_value = self.current_price.value * self.quantity.value
        if self.side == PositionSide.SHORT:
            # For short positions, market value represents the liability
            market_value = -market_value
        
        return Money(market_value)
    
    def get_unrealized_pnl(self) -> Optional[Money]:
        """Calculate unrealized profit/loss."""
        if not self.current_price or not self.cost_basis:
            return None
        
        market_value = self.get_market_value()
        if not market_value:
            return None
        
        if self.side == PositionSide.LONG:
            return Money(market_value.amount - self.cost_basis.amount)
        else:  # SHORT
            return Money(self.cost_basis.amount - market_value.amount)
    
    def get_unrealized_pnl_percent(self) -> Optional[float]:
        """Calculate unrealized P&L as percentage."""
        unrealized_pnl = self.get_unrealized_pnl()
        if not unrealized_pnl or not self.cost_basis or self.cost_basis.amount == 0:
            return None
        
        return float((unrealized_pnl.amount / abs(self.cost_basis.amount)) * 100)
    
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == PositionSide.LONG
    
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == PositionSide.SHORT
    
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.side == PositionSide.FLAT or self.quantity.value == 0
    
    def close_position(self, close_price: Price, close_time: datetime = None) -> None:
        """Close the position."""
        self.side = PositionSide.FLAT
        self.quantity = Quantity(0)
        self.current_price = close_price
        if close_time:
            self.mark_updated()
    
    def _update_position_metrics(self) -> None:
        """Update position metrics based on trades."""
        if not self.trades:
            return
        
        total_cost = Decimal('0')
        total_quantity = 0
        
        for trade in self.trades:
            if trade.side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
                total_cost += trade.execution.execution_price.value * trade.quantity.value
                total_quantity += trade.quantity.value
            else:  # SELL or SELL_SHORT
                total_cost -= trade.execution.execution_price.value * trade.quantity.value
                total_quantity -= trade.quantity.value
        
        self.quantity = Quantity(abs(total_quantity))
        if total_quantity > 0:
            self.side = PositionSide.LONG
        elif total_quantity < 0:
            self.side = PositionSide.SHORT
        else:
            self.side = PositionSide.FLAT
        
        if self.quantity.value > 0:
            self.average_price = Price(abs(total_cost) / self.quantity.value)
            self.cost_basis = Money(abs(total_cost))


@dataclass 
class Trade(Entity):
    """
    Represents an executed trade.
    
    A trade is the result of an order execution.
    """
    
    id: EntityId
    order_id: OrderId
    position_id: Optional[PositionId]
    ticker_id: TickerId
    side: OrderSide
    quantity: Quantity
    execution: TradeExecution
    strategy_id: Optional[str] = None
    
    def validate(self) -> None:
        if not self.order_id:
            raise ValueError("Order ID is required")
        
        if not self.ticker_id:
            raise ValueError("Ticker ID is required")
        
        if not self.execution:
            raise ValueError("Trade execution is required")
        
        if self.quantity.value != self.execution.execution_quantity.value:
            raise ValueError("Trade quantity must match execution quantity")
    
    def get_gross_amount(self) -> Money:
        """Get gross trade amount (price * quantity)."""
        amount = self.execution.execution_price.value * self.quantity.value
        return Money(amount)
    
    def get_net_amount(self) -> Money:
        """Get net trade amount including fees and commissions."""
        gross = self.get_gross_amount()
        return gross.subtract(self.execution.commission).subtract(self.execution.fees)


@dataclass
class Portfolio(AggregateRoot):
    """
    Represents a trading portfolio.
    
    A portfolio is a collection of positions and cash.
    """
    
    id: PortfolioId
    name: str
    account_id: EntityId
    cash_balance: Money
    initial_value: Money
    positions: Dict[str, Position] = field(default_factory=dict)
    risk_parameters: Optional[RiskParameters] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def validate(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Portfolio name is required")
        
        if not self.account_id:
            raise ValueError("Account ID is required")
        
        if self.cash_balance.amount < 0:
            raise ValueError("Cash balance cannot be negative")
    
    def add_position(self, position: Position) -> None:
        """Add a position to the portfolio."""
        if position.portfolio_id != self.id:
            raise ValueError("Position must belong to this portfolio")
        
        position_key = str(position.ticker_id.value)
        self.positions[position_key] = position
        self.mark_updated()
    
    def remove_position(self, ticker_id: TickerId) -> None:
        """Remove a position from the portfolio."""
        position_key = str(ticker_id.value)
        if position_key in self.positions:
            del self.positions[position_key]
            self.mark_updated()
    
    def get_position(self, ticker_id: TickerId) -> Optional[Position]:
        """Get a position by ticker ID."""
        position_key = str(ticker_id.value)
        return self.positions.get(position_key)
    
    def update_cash_balance(self, amount: Money) -> None:
        """Update cash balance (positive for deposits, negative for withdrawals)."""
        new_balance = self.cash_balance.add(amount)
        if new_balance.amount < 0:
            raise ValueError("Insufficient cash balance")
        
        self.cash_balance = new_balance
        self.mark_updated()
    
    def get_total_equity_value(self) -> Money:
        """Calculate total value of all equity positions."""
        total = Decimal('0')
        
        for position in self.positions.values():
            if not position.is_flat():
                market_value = position.get_market_value()
                if market_value:
                    total += market_value.amount
        
        return Money(total, self.cash_balance.currency)
    
    def get_total_portfolio_value(self) -> Money:
        """Calculate total portfolio value (cash + equity)."""
        equity_value = self.get_total_equity_value()
        return self.cash_balance.add(equity_value)
    
    def get_total_unrealized_pnl(self) -> Money:
        """Calculate total unrealized P&L across all positions."""
        total_pnl = Decimal('0')
        
        for position in self.positions.values():
            if not position.is_flat():
                pnl = position.get_unrealized_pnl()
                if pnl:
                    total_pnl += pnl.amount
        
        return Money(total_pnl, self.cash_balance.currency)
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Get comprehensive portfolio metrics."""
        total_value = self.get_total_portfolio_value()
        equity_value = self.get_total_equity_value()
        unrealized_pnl = self.get_total_unrealized_pnl()
        
        # Calculate total return
        if self.initial_value.amount != 0:
            total_return_pct = ((total_value.amount - self.initial_value.amount) / self.initial_value.amount) * 100
        else:
            total_return_pct = Decimal('0')
        
        from ..shared.value_objects import Percentage
        
        return PortfolioMetrics(
            total_value=total_value,
            cash_balance=self.cash_balance,
            equity_value=equity_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=Money(Decimal('0')),  # Would need trade history to calculate
            total_return=Percentage(total_return_pct),
            daily_return=Percentage(Decimal('0')),  # Would need daily tracking
            max_drawdown=Percentage(Decimal('0'))   # Would need historical data
        )


@dataclass
class Account(AggregateRoot):
    """
    Represents a trading account.
    
    An account can contain multiple portfolios.
    """
    
    id: EntityId
    account_number: str
    account_type: str  # 'CASH', 'MARGIN', 'IRA', etc.
    owner_name: str
    portfolios: List[PortfolioId] = field(default_factory=list)
    is_active: bool = True
    opened_date: date = field(default_factory=date.today)
    
    def validate(self) -> None:
        if not self.account_number or not self.account_number.strip():
            raise ValueError("Account number is required")
        
        if not self.owner_name or not self.owner_name.strip():
            raise ValueError("Owner name is required")
        
        valid_types = ['CASH', 'MARGIN', 'IRA', 'ROTH_IRA', '401K']
        if self.account_type not in valid_types:
            raise ValueError(f"Account type must be one of: {valid_types}")
    
    def add_portfolio(self, portfolio_id: PortfolioId) -> None:
        """Add a portfolio to this account."""
        if portfolio_id not in self.portfolios:
            self.portfolios.append(portfolio_id)
            self.mark_updated()
    
    def remove_portfolio(self, portfolio_id: PortfolioId) -> None:
        """Remove a portfolio from this account."""
        if portfolio_id in self.portfolios:
            self.portfolios.remove(portfolio_id)
            self.mark_updated()
    
    def close_account(self, close_date: date = None) -> None:
        """Close the account."""
        self.is_active = False
        if close_date:
            # Would set close date if we had that field
            pass
        self.mark_updated()