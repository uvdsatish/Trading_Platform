"""
Configuration service with dependency injection support.
Provides centralized access to all configuration components.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging
from abc import ABC, abstractmethod

from .environment import EnvironmentConfig
from .database import DatabaseConfig, MultiDatabaseConfig
from .providers import DataProviderConfig


class IConfigurationService(ABC):
    """Abstract interface for configuration service."""
    
    @abstractmethod
    def get_database_config(self, name: Optional[str] = None) -> DatabaseConfig:
        """Get database configuration."""
        pass
    
    @abstractmethod
    def get_data_provider_config(self, name: Optional[str] = None) -> Any:
        """Get data provider configuration."""
        pass
    
    @abstractmethod
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get general configuration setting."""
        pass
    
    @abstractmethod
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if feature is enabled."""
        pass


class ConfigurationService(IConfigurationService):
    """
    Main configuration service that provides unified access to all configurations.
    This is the primary entry point for configuration in the application.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for configuration service."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, environment: Optional[str] = None, config_dir: Optional[Path] = None):
        """
        Initialize configuration service.
        
        Args:
            environment: Environment name (dev, test, staging, prod)
            config_dir: Directory containing configuration files
        """
        # Skip re-initialization for singleton
        if hasattr(self, '_initialized'):
            return
        
        self.logger = logging.getLogger(__name__)
        
        # Load environment configuration
        self.env_config = EnvironmentConfig(environment, config_dir)
        self.environment = self.env_config.environment
        
        # Initialize database configurations
        self._init_database_configs()
        
        # Initialize data provider configurations
        self._init_provider_configs()
        
        self._initialized = True
        
        self.logger.info(f"Configuration service initialized for environment: {self.environment}")
    
    def _init_database_configs(self) -> None:
        """Initialize database configurations."""
        db_configs = self.env_config.get('database', {})
        
        if not db_configs:
            self.logger.warning("No database configurations found")
            self.db_config = None
            return
        
        # Handle multiple database configurations
        if 'primary' in db_configs or 'secondary' in db_configs:
            # Multiple databases configured
            self.db_config = MultiDatabaseConfig(db_configs)
        else:
            # Single database configured
            self.db_config = DatabaseConfig(db_configs)
    
    def _init_provider_configs(self) -> None:
        """Initialize data provider configurations."""
        provider_configs = self.env_config.get('data_providers', {})
        
        if not provider_configs:
            self.logger.warning("No data provider configurations found")
            self.provider_config = None
            return
        
        self.provider_config = DataProviderConfig(provider_configs)
    
    def get_database_config(self, name: Optional[str] = None) -> DatabaseConfig:
        """
        Get database configuration.
        
        Args:
            name: Database name for multi-database setup
            
        Returns:
            Database configuration
        """
        if self.db_config is None:
            raise ValueError("No database configuration available")
        
        if isinstance(self.db_config, MultiDatabaseConfig):
            return self.db_config.get_database(name) if name else self.db_config.get_primary()
        
        return self.db_config
    
    def get_database_connection_string(self, name: Optional[str] = None, 
                                      include_password: bool = True) -> str:
        """
        Get database connection string.
        
        Args:
            name: Database name for multi-database setup
            include_password: Whether to include password
            
        Returns:
            Database connection string
        """
        db_config = self.get_database_config(name)
        return db_config.get_connection_string(include_password)
    
    def get_data_provider_config(self, name: Optional[str] = None) -> Any:
        """
        Get data provider configuration.
        
        Args:
            name: Provider name
            
        Returns:
            Data provider configuration
        """
        if self.provider_config is None:
            raise ValueError("No data provider configuration available")
        
        return self.provider_config.get_provider(name)
    
    def get_iqfeed_config(self) -> Dict[str, Any]:
        """
        Get IQFeed configuration (backward compatibility).
        
        Returns:
            IQFeed configuration dictionary
        """
        iqfeed = self.get_data_provider_config('iqfeed')
        return iqfeed.to_dict() if iqfeed else {}
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get general configuration setting.
        
        Args:
            key: Setting key (supports dot notation)
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        return self.env_config.get(key, default)
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if feature is enabled.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            True if feature is enabled
        """
        return self.env_config.get_feature_flag(feature_name)
    
    def get_log_level(self) -> str:
        """Get configured log level."""
        return self.get_setting('logging.level', 'INFO')
    
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """
        Get API configuration.
        
        Args:
            api_name: Name of the API
            
        Returns:
            API configuration dictionary
        """
        return self.get_setting(f'apis.{api_name}', {})
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.env_config.is_development()
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env_config.is_production()
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.env_config.is_testing()
    
    def reload(self) -> None:
        """Reload configuration from sources."""
        self.env_config = EnvironmentConfig(self.environment)
        self._init_database_configs()
        self._init_provider_configs()
        self.logger.info("Configuration reloaded")
    
    def to_dict(self, safe: bool = True) -> Dict[str, Any]:
        """
        Export configuration as dictionary.
        
        Args:
            safe: If True, mask sensitive information
            
        Returns:
            Configuration dictionary
        """
        config = self.env_config.to_dict()
        
        if safe:
            # Mask sensitive information
            if 'database' in config:
                for db_name, db_config in config['database'].items():
                    if isinstance(db_config, dict) and 'password' in db_config:
                        db_config['password'] = '***'
            
            if 'data_providers' in config:
                for provider_name, provider_config in config['data_providers'].items():
                    if isinstance(provider_config, dict):
                        if 'api_key' in provider_config:
                            provider_config['api_key'] = '***'
                        if 'credentials' in provider_config:
                            provider_config['credentials']['password'] = '***'
        
        return config


# Convenience function for getting configuration service instance
def get_config_service(environment: Optional[str] = None) -> ConfigurationService:
    """
    Get configuration service instance.
    
    Args:
        environment: Optional environment override
        
    Returns:
        Configuration service instance
    """
    return ConfigurationService(environment)