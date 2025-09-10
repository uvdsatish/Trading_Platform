"""
Base configuration module for the trading platform.
Provides abstract base classes and core configuration functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import yaml
from pathlib import Path


class BaseConfig(ABC):
    """Abstract base class for all configuration types."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration with optional dictionary.
        
        Args:
            config_dict: Optional configuration dictionary
        """
        self._config = config_dict or {}
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the configuration. Must be implemented by subclasses."""
        pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key with optional default.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()


class ConfigLoader:
    """Utility class for loading configuration from various sources."""
    
    @staticmethod
    def load_from_yaml(file_path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            file_path: Path to YAML configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    @staticmethod
    def load_from_env(prefix: str = "TRADING_") -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Prefix for environment variables to load
            
        Returns:
            Configuration dictionary from environment
        """
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                
                # Convert underscores to nested dict notation
                keys = config_key.split('_')
                current = config
                
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                
                # Try to parse value as appropriate type
                current[keys[-1]] = ConfigLoader._parse_env_value(value)
        
        return config
    
    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """
        Parse environment variable value to appropriate type.
        
        Args:
            value: String value from environment
            
        Returns:
            Parsed value (bool, int, float, or string)
        """
        # Boolean values
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # Default to string
        return value


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass