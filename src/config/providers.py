"""
Data provider configuration management.
Handles configuration for IQFeed and other market data providers.
"""

from typing import Dict, Any, Optional
from .base import BaseConfig, ConfigurationError


class IQFeedConfig(BaseConfig):
    """Configuration for IQFeed data provider."""
    
    DEFAULT_HOST = '127.0.0.1'
    DEFAULT_PORTS = {
        'level1': 5009,
        'lookup': 9100,
        'level2': 9200,
        'admin': 9300,
        'derivative': 9400
    }
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize IQFeed configuration.
        
        Args:
            config_dict: IQFeed configuration dictionary
        """
        super().__init__(config_dict)
        self._set_defaults()
    
    def _set_defaults(self) -> None:
        """Set default values for IQFeed configuration."""
        if not self.get('host'):
            self.set('host', self.DEFAULT_HOST)
        
        # Set default ports if not specified
        for service, default_port in self.DEFAULT_PORTS.items():
            if not self.get(f'ports.{service}'):
                self.set(f'ports.{service}', default_port)
    
    def _validate_config(self) -> None:
        """Validate IQFeed configuration."""
        # Validate credentials if provided
        if self.get('credentials.username') and not self.get('credentials.password'):
            raise ConfigurationError("IQFeed password required when username is provided")
        
        # Validate ports are integers
        ports = self.get('ports', {})
        for service, port in ports.items():
            if not isinstance(port, int) or port < 1 or port > 65535:
                raise ConfigurationError(f"Invalid port for IQFeed {service}: {port}")
        
        # Validate product ID if required
        if self.get('requires_product_id', False) and not self.get('product_id'):
            raise ConfigurationError("IQFeed product ID is required")
    
    def get_connection_params(self, service: str = 'lookup') -> Dict[str, Any]:
        """
        Get connection parameters for specific IQFeed service.
        
        Args:
            service: IQFeed service type (lookup, level1, level2, admin, derivative)
            
        Returns:
            Connection parameters dictionary
        """
        return {
            'host': self.get('host'),
            'port': self.get(f'ports.{service}'),
            'timeout': self.get('timeout', 60)
        }
    
    def get_credentials(self) -> Optional[Dict[str, str]]:
        """
        Get IQFeed credentials.
        
        Returns:
            Credentials dictionary or None if not configured
        """
        username = self.get('credentials.username')
        password = self.get('credentials.password')
        
        if username and password:
            return {
                'username': username,
                'password': password,
                'product_id': self.get('product_id', ''),
                'product_version': self.get('product_version', '1.0')
            }
        return None
    
    def get_data_settings(self) -> Dict[str, Any]:
        """
        Get data retrieval settings.
        
        Returns:
            Data settings dictionary
        """
        return {
            'max_symbols': self.get('limits.max_symbols', 500),
            'max_days': self.get('limits.max_days', 365),
            'chunk_size': self.get('limits.chunk_size', 100),
            'retry_attempts': self.get('retry.attempts', 3),
            'retry_delay': self.get('retry.delay', 5),
            'update_interval': self.get('update_interval', 60),
            'market_hours_only': self.get('market_hours_only', True)
        }
    
    def is_enabled(self) -> bool:
        """Check if IQFeed is enabled."""
        return self.get('enabled', True)


class AlphaVantageConfig(BaseConfig):
    """Configuration for Alpha Vantage data provider."""
    
    BASE_URL = 'https://www.alphavantage.co/query'
    
    def _validate_config(self) -> None:
        """Validate Alpha Vantage configuration."""
        if not self.get('api_key'):
            raise ConfigurationError("Alpha Vantage API key is required")
        
        # Validate rate limits
        rate_limit = self.get('rate_limit', 5)
        if not isinstance(rate_limit, int) or rate_limit < 1:
            raise ConfigurationError(f"Invalid rate limit: {rate_limit}")
    
    def get_api_params(self) -> Dict[str, str]:
        """Get API request parameters."""
        return {
            'apikey': self.get('api_key'),
            'datatype': self.get('data_format', 'json')
        }
    
    def get_rate_limit(self) -> int:
        """Get API rate limit (calls per minute)."""
        return self.get('rate_limit', 5)


class YahooFinanceConfig(BaseConfig):
    """Configuration for Yahoo Finance data provider."""
    
    def _validate_config(self) -> None:
        """Validate Yahoo Finance configuration."""
        # Yahoo Finance doesn't require API keys
        pass
    
    def get_download_settings(self) -> Dict[str, Any]:
        """Get download settings."""
        return {
            'threads': self.get('threads', 5),
            'chunk_size': self.get('chunk_size', 100),
            'timeout': self.get('timeout', 30),
            'retry_attempts': self.get('retry.attempts', 3)
        }


class DataProviderConfig:
    """Unified configuration for all data providers."""
    
    PROVIDER_CLASSES = {
        'iqfeed': IQFeedConfig,
        'alphavantage': AlphaVantageConfig,
        'yahoo': YahooFinanceConfig
    }
    
    def __init__(self, providers_config: Dict[str, Dict[str, Any]]):
        """
        Initialize data provider configuration.
        
        Args:
            providers_config: Dictionary of provider configurations
        """
        self.providers = {}
        self.primary_provider = None
        
        for name, config in providers_config.items():
            provider_type = config.get('type', name).lower()
            
            if provider_type not in self.PROVIDER_CLASSES:
                raise ConfigurationError(f"Unknown data provider type: {provider_type}")
            
            provider_class = self.PROVIDER_CLASSES[provider_type]
            self.providers[name] = provider_class(config)
            
            # Set primary provider
            if config.get('primary', False) or len(self.providers) == 1:
                self.primary_provider = name
    
    def get_provider(self, name: Optional[str] = None) -> BaseConfig:
        """
        Get data provider configuration.
        
        Args:
            name: Provider name (uses primary if not specified)
            
        Returns:
            Provider configuration
        """
        if name is None:
            name = self.primary_provider
        
        if name not in self.providers:
            raise ConfigurationError(f"Data provider '{name}' not found")
        
        return self.providers[name]
    
    def get_active_providers(self) -> Dict[str, BaseConfig]:
        """Get all enabled data providers."""
        return {
            name: provider 
            for name, provider in self.providers.items()
            if provider.get('enabled', True)
        }