"""
Environment-specific configuration management.
Handles development, testing, staging, and production configurations.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging
from .base import BaseConfig, ConfigLoader, ConfigurationError


class EnvironmentConfig(BaseConfig):
    """
    Environment-aware configuration that merges settings from multiple sources.
    Priority order (highest to lowest):
    1. Environment variables
    2. Environment-specific config file
    3. Default config file
    """
    
    VALID_ENVIRONMENTS = {'development', 'testing', 'staging', 'production'}
    
    def __init__(self, environment: Optional[str] = None, config_dir: Optional[Path] = None):
        """
        Initialize environment configuration.
        
        Args:
            environment: Environment name (dev, test, staging, prod)
            config_dir: Directory containing configuration files
        """
        self.environment = self._determine_environment(environment)
        self.config_dir = config_dir or self._get_default_config_dir()
        
        # Load and merge configurations
        config = self._load_merged_config()
        super().__init__(config)
        
        # Set up logging
        self._configure_logging()
    
    def _determine_environment(self, environment: Optional[str]) -> str:
        """
        Determine the current environment.
        
        Args:
            environment: Explicitly provided environment
            
        Returns:
            Environment name
        """
        if environment:
            env = environment.lower()
        else:
            # Check environment variable
            import os
            env = os.getenv('TRADING_ENV', 'development').lower()
        
        if env not in self.VALID_ENVIRONMENTS:
            raise ConfigurationError(
                f"Invalid environment: {env}. Must be one of {self.VALID_ENVIRONMENTS}"
            )
        
        return env
    
    def _get_default_config_dir(self) -> Path:
        """Get the default configuration directory."""
        # Assuming config files are in project_root/config
        from pathlib import Path
        return Path(__file__).parent.parent.parent / 'config'
    
    def _load_merged_config(self) -> Dict[str, Any]:
        """
        Load and merge configuration from all sources.
        
        Returns:
            Merged configuration dictionary
        """
        config = {}
        
        # 1. Load default configuration
        default_config_file = self.config_dir / 'application.yaml'
        if default_config_file.exists():
            config = ConfigLoader.load_from_yaml(default_config_file)
        
        # 2. Load environment-specific configuration
        env_config_file = self.config_dir / f'application-{self.environment}.yaml'
        if env_config_file.exists():
            env_config = ConfigLoader.load_from_yaml(env_config_file)
            config = self._deep_merge(config, env_config)
        
        # 3. Override with environment variables
        env_vars = ConfigLoader.load_from_env()
        config = self._deep_merge(config, env_vars)
        
        # Add environment to config
        config['environment'] = self.environment
        
        return config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Dictionary with values to override
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self) -> None:
        """Validate the configuration based on environment."""
        # In production, certain settings are required
        if self.environment == 'production':
            required_keys = [
                'database.host',
                'database.port',
                'database.name',
                'database.user',
                'security.secret_key'
            ]
            
            for key in required_keys:
                if not self.get(key):
                    raise ConfigurationError(
                        f"Required configuration key '{key}' not found for production environment"
                    )
    
    def _configure_logging(self) -> None:
        """Configure logging based on environment settings."""
        log_level = self.get('logging.level', 'INFO')
        log_format = self.get(
            'logging.format',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Different log levels for different environments
        if self.environment == 'development':
            log_level = self.get('logging.level', 'DEBUG')
        elif self.environment == 'production':
            log_level = self.get('logging.level', 'WARNING')
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format
        )
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == 'development'
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == 'production'
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == 'testing'
    
    def get_feature_flag(self, feature_name: str, default: bool = False) -> bool:
        """
        Get feature flag value.
        
        Args:
            feature_name: Name of the feature
            default: Default value if not found
            
        Returns:
            Feature flag value
        """
        return self.get(f'features.{feature_name}', default)