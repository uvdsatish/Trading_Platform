"""
Trading value objects.
"""

from dataclasses import dataclass
from enum import Enum
from decimal import Decimal
from datetime import datetime
from typing import Optional

from ..shared.value_objects import ValueObject, Money, Price, Quantity, Percentage


class OrderType(Enum):
    """Types of trading orders."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderSide(Enum):
    """Order side (buy/sell)."""
    BUY = "BUY"
    SELL = "SELL"
    SELL_SHORT = "SELL_SHORT"
    BUY_TO_COVER = "BUY_TO_COVER"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(Enum):
    """Time in force for orders."""
    DAY = "DAY"              # Good for day
    GTC = "GTC"              # Good till cancelled
    IOC = "IOC"              # Immediate or cancel
    FOK = "FOK"              # Fill or kill
    GTD = "GTD"              # Good till date


class PositionSide(Enum):
    """Position side (long/short)."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass(frozen=True)
class TradeExecution(ValueObject):
    """Represents a trade execution."""
    
    execution_price: Price
    execution_quantity: Quantity
    execution_time: datetime
    commission: Money
    fees: Money
    execution_id: str
    
    def validate(self) -> None:
        if not self.execution_price:
            raise ValueError("Execution price is required")
        if not self.execution_quantity:
            raise ValueError("Execution quantity is required")
        if not self.execution_time:
            raise ValueError("Execution time is required")
        if not self.execution_id:
            raise ValueError("Execution ID is required")
    
    def get_total_cost(self) -> Money:
        """Calculate total cost including commissions and fees."""
        gross_amount = Money(
            self.execution_price.value * self.execution_quantity.value,
            self.commission.currency
        )
        return gross_amount.add(self.commission).add(self.fees)


@dataclass(frozen=True)
class RiskParameters(ValueObject):
    """Risk management parameters."""
    
    max_position_size: Optional[Money] = None
    max_portfolio_exposure: Optional[Percentage] = None
    stop_loss_percentage: Optional[Percentage] = None
    take_profit_percentage: Optional[Percentage] = None
    max_daily_loss: Optional[Money] = None
    max_drawdown: Optional[Percentage] = None
    
    def validate(self) -> None:
        if self.stop_loss_percentage and self.stop_loss_percentage.value <= 0:
            raise ValueError("Stop loss percentage must be positive")
        
        if self.take_profit_percentage and self.take_profit_percentage.value <= 0:
            raise ValueError("Take profit percentage must be positive")
        
        if self.max_portfolio_exposure and (
            self.max_portfolio_exposure.value <= 0 or 
            self.max_portfolio_exposure.value > 100
        ):
            raise ValueError("Portfolio exposure must be between 0 and 100%")
    
    def calculate_stop_loss_price(self, entry_price: Price, position_side: PositionSide) -> Optional[Price]:
        """Calculate stop loss price based on parameters."""
        if not self.stop_loss_percentage:
            return None
        
        if position_side == PositionSide.LONG:
            # For long positions, stop loss is below entry price
            stop_price = entry_price.value * (1 - self.stop_loss_percentage.to_decimal())
        else:
            # For short positions, stop loss is above entry price
            stop_price = entry_price.value * (1 + self.stop_loss_percentage.to_decimal())
        
        return Price(stop_price)
    
    def calculate_take_profit_price(self, entry_price: Price, position_side: PositionSide) -> Optional[Price]:
        """Calculate take profit price based on parameters."""
        if not self.take_profit_percentage:
            return None
        
        if position_side == PositionSide.LONG:
            # For long positions, take profit is above entry price
            target_price = entry_price.value * (1 + self.take_profit_percentage.to_decimal())
        else:
            # For short positions, take profit is below entry price
            target_price = entry_price.value * (1 - self.take_profit_percentage.to_decimal())
        
        return Price(target_price)


@dataclass(frozen=True)
class PortfolioMetrics(ValueObject):
    """Portfolio performance metrics."""
    
    total_value: Money
    cash_balance: Money
    equity_value: Money
    unrealized_pnl: Money
    realized_pnl: Money
    total_return: Percentage
    daily_return: Percentage
    max_drawdown: Percentage
    sharpe_ratio: Optional[float] = None
    beta: Optional[float] = None
    
    def validate(self) -> None:
        # Total value should equal cash + equity
        expected_total = self.cash_balance.add(self.equity_value)
        if abs(self.total_value.amount - expected_total.amount) > Decimal('0.01'):
            raise ValueError("Total value must equal cash balance plus equity value")
        
        if self.total_value.amount < 0:
            raise ValueError("Total portfolio value cannot be negative")
    
    def get_net_pnl(self) -> Money:
        """Get net profit/loss (realized + unrealized)."""
        return self.realized_pnl.add(self.unrealized_pnl)
    
    def get_equity_percentage(self) -> Percentage:
        """Get percentage of portfolio in equity positions."""
        if self.total_value.amount == 0:
            return Percentage(Decimal('0'))
        
        equity_pct = (self.equity_value.amount / self.total_value.amount) * 100
        return Percentage(equity_pct)
    
    def get_cash_percentage(self) -> Percentage:
        """Get percentage of portfolio in cash."""
        if self.total_value.amount == 0:
            return Percentage(Decimal('100'))
        
        cash_pct = (self.cash_balance.amount / self.total_value.amount) * 100
        return Percentage(cash_pct)


@dataclass(frozen=True)
class OrderConstraints(ValueObject):
    """Order validation constraints."""
    
    min_quantity: Optional[Quantity] = None
    max_quantity: Optional[Quantity] = None
    min_price: Optional[Price] = None
    max_price: Optional[Price] = None
    tick_size: Optional[Price] = None
    lot_size: Optional[Quantity] = None
    
    def validate(self) -> None:
        if self.min_quantity and self.max_quantity:
            if self.min_quantity.value > self.max_quantity.value:
                raise ValueError("Min quantity cannot exceed max quantity")
        
        if self.min_price and self.max_price:
            if self.min_price.value > self.max_price.value:
                raise ValueError("Min price cannot exceed max price")
    
    def validate_order(self, quantity: Quantity, price: Optional[Price] = None) -> None:
        """Validate order against constraints."""
        if self.min_quantity and quantity.value < self.min_quantity.value:
            raise ValueError(f"Quantity {quantity.value} below minimum {self.min_quantity.value}")
        
        if self.max_quantity and quantity.value > self.max_quantity.value:
            raise ValueError(f"Quantity {quantity.value} exceeds maximum {self.max_quantity.value}")
        
        if self.lot_size and quantity.value % self.lot_size.value != 0:
            raise ValueError(f"Quantity must be multiple of lot size {self.lot_size.value}")
        
        if price:
            if self.min_price and price.value < self.min_price.value:
                raise ValueError(f"Price {price.value} below minimum {self.min_price.value}")
            
            if self.max_price and price.value > self.max_price.value:
                raise ValueError(f"Price {price.value} exceeds maximum {self.max_price.value}")
            
            if self.tick_size:
                remainder = price.value % self.tick_size.value
                if remainder != 0:
                    raise ValueError(f"Price must be multiple of tick size {self.tick_size.value}")


@dataclass(frozen=True)
class PositionMetrics(ValueObject):
    """Position-level metrics."""
    
    market_value: Money
    unrealized_pnl: Money
    unrealized_pnl_percent: Percentage
    cost_basis: Money
    average_price: Price
    duration_days: int
    
    def validate(self) -> None:
        if self.duration_days < 0:
            raise ValueError("Duration days cannot be negative")
    
    def is_profitable(self) -> bool:
        """Check if position is profitable."""
        return self.unrealized_pnl.amount > 0
    
    def is_at_loss(self) -> bool:
        """Check if position is at a loss."""
        return self.unrealized_pnl.amount < 0