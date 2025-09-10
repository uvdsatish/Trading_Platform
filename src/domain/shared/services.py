"""
Domain services interface.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar('T')


class DomainService(ABC, Generic[T]):
    """
    Base class for domain services.
    
    Domain services contain business logic that doesn't naturally fit within
    an entity or value object. They are stateless and focus on domain operations.
    """
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> T:
        """Execute the domain service operation."""
        pass