"""
Configuration management module for the trading platform.

This module provides centralized configuration management with:
- Environment-aware configuration (dev, test, staging, prod)
- Multiple database support
- Data provider configurations (IQFeed, Alpha Vantage, Yahoo)
- Feature flags
- Dependency injection support

Usage:
    from src.config import ConfigurationService, get_config_service
    
    # Get configuration service instance
    config = get_config_service()
    
    # Get database configuration
    db_config = config.get_database_config('technical_analysis')
    conn_string = config.get_database_connection_string('technical_analysis')
    
    # Get data provider configuration
    iqfeed_config = config.get_data_provider_config('iqfeed')
    
    # Check feature flags
    if config.is_feature_enabled('backtesting'):
        # Run backtesting logic
        pass
"""

from .base import BaseConfig, ConfigLoader, ConfigurationError
from .environment import EnvironmentConfig
from .database import DatabaseConfig, MultiDatabaseConfig
from .providers import (
    IQFeedConfig,
    AlphaVantageConfig,
    YahooFinanceConfig,
    DataProviderConfig
)
from .service import ConfigurationService, IConfigurationService, get_config_service

__all__ = [
    # Base classes
    'BaseConfig',
    'ConfigLoader',
    'ConfigurationError',
    
    # Environment
    'EnvironmentConfig',
    
    # Database
    'DatabaseConfig',
    'MultiDatabaseConfig',
    
    # Data providers
    'IQFeedConfig',
    'AlphaVantageConfig',
    'YahooFinanceConfig',
    'DataProviderConfig',
    
    # Service
    'ConfigurationService',
    'IConfigurationService',
    'get_config_service',
]