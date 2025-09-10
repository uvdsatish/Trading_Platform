"""
Database configuration management.
Handles database connection settings and connection string generation.
"""

from typing import Dict, Any, Optional
from urllib.parse import quote_plus
from .base import BaseConfig, ConfigurationError


class DatabaseConfig(BaseConfig):
    """Configuration for database connections."""
    
    REQUIRED_FIELDS = {'host', 'port', 'name', 'user', 'password'}
    DEFAULT_PORTS = {
        'postgresql': 5432,
        'mysql': 3306,
        'sqlite': None
    }
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize database configuration.
        
        Args:
            config_dict: Database configuration dictionary
        """
        super().__init__(config_dict)
        self.db_type = self._config.get('type', 'postgresql')
    
    def _validate_config(self) -> None:
        """Validate database configuration."""
        if self.db_type == 'sqlite':
            # SQLite only needs a path
            if not self.get('path'):
                raise ConfigurationError("SQLite database requires 'path' field")
            return
        
        # For other databases, check required fields
        missing_fields = []
        for field in self.REQUIRED_FIELDS:
            if not self.get(field):
                missing_fields.append(field)
        
        if missing_fields:
            raise ConfigurationError(
                f"Missing required database configuration fields: {', '.join(missing_fields)}"
            )
        
        # Validate port is a number
        port = self.get('port')
        if port and not isinstance(port, int):
            try:
                self.set('port', int(port))
            except ValueError:
                raise ConfigurationError(f"Invalid port number: {port}")
    
    def get_connection_string(self, include_password: bool = True) -> str:
        """
        Generate database connection string.
        
        Args:
            include_password: Whether to include password in connection string
            
        Returns:
            Database connection string
        """
        if self.db_type == 'sqlite':
            return f"sqlite:///{self.get('path')}"
        
        # Build connection string for other databases
        user = self.get('user')
        password = self.get('password') if include_password else '***'
        host = self.get('host')
        port = self.get('port', self.DEFAULT_PORTS.get(self.db_type))
        database = self.get('name')
        
        # URL encode password to handle special characters
        if include_password and password:
            password = quote_plus(password)
        
        # Build base connection string
        if self.db_type == 'postgresql':
            conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        elif self.db_type == 'mysql':
            conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        else:
            raise ConfigurationError(f"Unsupported database type: {self.db_type}")
        
        # Add additional parameters
        params = self.get_connection_params()
        if params:
            param_str = '&'.join([f"{k}={v}" for k, v in params.items()])
            conn_str = f"{conn_str}?{param_str}"
        
        return conn_str
    
    def get_connection_params(self) -> Dict[str, Any]:
        """
        Get additional connection parameters.
        
        Returns:
            Dictionary of connection parameters
        """
        params = {}
        
        # SSL/TLS settings
        if self.get('ssl.enabled'):
            params['sslmode'] = self.get('ssl.mode', 'require')
            if self.get('ssl.ca_cert'):
                params['sslcert'] = self.get('ssl.ca_cert')
        
        # Connection pool settings
        if self.get('pool.size'):
            params['pool_size'] = self.get('pool.size')
        if self.get('pool.max_overflow'):
            params['max_overflow'] = self.get('pool.max_overflow')
        
        # Timeout settings
        if self.get('connect_timeout'):
            params['connect_timeout'] = self.get('connect_timeout')
        
        # Additional custom parameters
        custom_params = self.get('params', {})
        params.update(custom_params)
        
        return params
    
    def get_pool_config(self) -> Dict[str, Any]:
        """
        Get connection pool configuration.
        
        Returns:
            Connection pool configuration dictionary
        """
        return {
            'pool_size': self.get('pool.size', 5),
            'max_overflow': self.get('pool.max_overflow', 10),
            'pool_timeout': self.get('pool.timeout', 30),
            'pool_recycle': self.get('pool.recycle', 3600),
            'pool_pre_ping': self.get('pool.pre_ping', True)
        }
    
    def to_dict_safe(self) -> Dict[str, Any]:
        """
        Return configuration dictionary with password masked.
        
        Returns:
            Configuration dictionary with sensitive data masked
        """
        config = self.to_dict()
        if 'password' in config:
            config['password'] = '***'
        return config


class MultiDatabaseConfig:
    """Configuration for multiple database connections."""
    
    def __init__(self, configs: Dict[str, Dict[str, Any]]):
        """
        Initialize multi-database configuration.
        
        Args:
            configs: Dictionary of database configurations by name
        """
        self.databases = {}
        for name, config in configs.items():
            self.databases[name] = DatabaseConfig(config)
    
    def get_database(self, name: str) -> DatabaseConfig:
        """
        Get database configuration by name.
        
        Args:
            name: Database configuration name
            
        Returns:
            Database configuration
            
        Raises:
            ConfigurationError: If database not found
        """
        if name not in self.databases:
            raise ConfigurationError(f"Database configuration '{name}' not found")
        return self.databases[name]
    
    def get_primary(self) -> DatabaseConfig:
        """Get primary database configuration."""
        return self.get_database('primary')
    
    def list_databases(self) -> list:
        """List available database configurations."""
        return list(self.databases.keys())