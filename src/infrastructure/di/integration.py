"""
Integration module for dependency injection with configuration and database systems.

Provides pre-configured service registrations and bootstrap functions
for setting up the dependency injection container with core services.
"""

from typing import Optional

from src.config import (
    ConfigurationService,
    DatabaseConfig,
    MultiDatabaseConfig,
    IQFeedConfig,
    DataProviderConfig,
    TradingConfig,
    BacktestingConfig
)
from src.infrastructure.database import (
    IConnectionPool,
    PostgreSQLConnectionPool,
    ISessionFactory,
    DatabaseSessionFactory,
    TransactionManager
)
from src.infrastructure.logging import (
    get_logger,
    TradingLogger,
    LoggerFactory
)
from .container import Container, ServiceLifetime
from .decorators import singleton, transient

logger = get_logger(__name__)


class ServiceBootstrapper:
    """Bootstraps the dependency injection container with core services."""
    
    @staticmethod
    def configure_core_services(container: Container) -> None:
        """
        Configure core infrastructure services.
        
        Args:
            container: DI container to configure
        """
        logger.debug("Configuring core services")
        
        # Configuration services
        container.register(
            ConfigurationService,
            ConfigurationService,
            ServiceLifetime.SINGLETON
        )
        
        # Logging services
        container.register_factory(
            TradingLogger,
            lambda: get_logger("trading"),
            ServiceLifetime.SINGLETON
        )
        
        container.register_instance(LoggerFactory, LoggerFactory())
        
        logger.debug("Core services configured")
    
    @staticmethod
    def configure_database_services(container: Container) -> None:
        """
        Configure database-related services.
        
        Args:
            container: DI container to configure
        """
        logger.debug("Configuring database services")
        
        # Register database configuration factory
        def create_database_config():
            config_service = container.resolve(ConfigurationService)
            return config_service.get_multi_database_config()
        
        container.register_factory(
            MultiDatabaseConfig,
            create_database_config,
            ServiceLifetime.SINGLETON
        )
        
        # Register connection pool factory
        def create_connection_pool():
            db_config = container.resolve(MultiDatabaseConfig)
            # Use the primary technical analysis database by default
            primary_config = db_config.get_database('technical_analysis')
            
            return PostgreSQLConnectionPool(
                host=primary_config.host,
                port=primary_config.port,
                database=primary_config.database,
                username=primary_config.username,
                password=primary_config.password,
                min_connections=primary_config.min_connections,
                max_connections=primary_config.max_connections,
                connection_timeout=primary_config.connection_timeout
            )
        
        container.register_factory(
            IConnectionPool,
            create_connection_pool,
            ServiceLifetime.SINGLETON
        )
        
        # Register session factory
        container.register(
            ISessionFactory,
            DatabaseSessionFactory,
            ServiceLifetime.SINGLETON
        )
        
        # Register transaction manager
        container.register(
            TransactionManager,
            TransactionManager,
            ServiceLifetime.SINGLETON
        )
        
        logger.debug("Database services configured")
    
    @staticmethod
    def configure_data_provider_services(container: Container) -> None:
        """
        Configure data provider services.
        
        Args:
            container: DI container to configure
        """
        logger.debug("Configuring data provider services")
        
        # Register IQFeed configuration factory
        def create_iqfeed_config():
            config_service = container.resolve(ConfigurationService)
            return config_service.get_iqfeed_config()
        
        container.register_factory(
            IQFeedConfig,
            create_iqfeed_config,
            ServiceLifetime.SINGLETON
        )
        
        # Register general data provider config factory
        def create_data_provider_config():
            config_service = container.resolve(ConfigurationService)
            return config_service.get_data_provider_config()
        
        container.register_factory(
            DataProviderConfig,
            create_data_provider_config,
            ServiceLifetime.SINGLETON
        )
        
        logger.debug("Data provider services configured")
    
    @staticmethod
    def configure_trading_services(container: Container) -> None:
        """
        Configure trading-related services.
        
        Args:
            container: DI container to configure
        """
        logger.debug("Configuring trading services")
        
        # Register trading configuration factory
        def create_trading_config():
            config_service = container.resolve(ConfigurationService)
            return config_service.get_trading_config()
        
        container.register_factory(
            TradingConfig,
            create_trading_config,
            ServiceLifetime.SINGLETON
        )
        
        # Register backtesting configuration factory
        def create_backtesting_config():
            config_service = container.resolve(ConfigurationService)
            return config_service.get_backtesting_config()
        
        container.register_factory(
            BacktestingConfig,
            create_backtesting_config,
            ServiceLifetime.SINGLETON
        )
        
        logger.debug("Trading services configured")
    
    @staticmethod
    def configure_domain_specific_databases(container: Container) -> None:
        """
        Configure domain-specific database connection pools.
        
        Args:
            container: DI container to configure
        """
        logger.debug("Configuring domain-specific database services")
        
        def create_domain_connection_pool(domain: str):
            """Create a connection pool for a specific domain."""
            def factory():
                db_config = container.resolve(MultiDatabaseConfig)
                domain_config = db_config.get_database(domain)
                
                return PostgreSQLConnectionPool(
                    host=domain_config.host,
                    port=domain_config.port,
                    database=domain_config.database,
                    username=domain_config.username,
                    password=domain_config.password,
                    min_connections=domain_config.min_connections,
                    max_connections=domain_config.max_connections,
                    connection_timeout=domain_config.connection_timeout
                )
            return factory
        
        # Register connection pools for different domains
        domains = [
            'technical_analysis',
            'internals',
            'fundamentals',
            'macro',
            'plurality',
            'backtesting'
        ]
        
        for domain in domains:
            # Create a unique type for each domain's connection pool
            pool_type_name = f"{domain.title().replace('_', '')}ConnectionPool"
            pool_type = type(pool_type_name, (IConnectionPool,), {})
            
            container.register_factory(
                pool_type,
                create_domain_connection_pool(domain),
                ServiceLifetime.SINGLETON
            )
            
            logger.debug(f"Registered connection pool for domain: {domain}")
        
        logger.debug("Domain-specific database services configured")


def create_configured_container() -> Container:
    """
    Create and configure a new container with all core services.
    
    Returns:
        Configured Container instance
    """
    container = Container()
    
    # Configure all service categories
    ServiceBootstrapper.configure_core_services(container)
    ServiceBootstrapper.configure_database_services(container)
    ServiceBootstrapper.configure_data_provider_services(container)
    ServiceBootstrapper.configure_trading_services(container)
    ServiceBootstrapper.configure_domain_specific_databases(container)
    
    logger.info("Dependency injection container configured with all core services")
    return container


def bootstrap_application(container: Optional[Container] = None) -> Container:
    """
    Bootstrap the application with dependency injection.
    
    Args:
        container: Optional existing container to configure
        
    Returns:
        Configured container
    """
    if container is None:
        container = create_configured_container()
    else:
        ServiceBootstrapper.configure_core_services(container)
        ServiceBootstrapper.configure_database_services(container)
        ServiceBootstrapper.configure_data_provider_services(container)
        ServiceBootstrapper.configure_trading_services(container)
        ServiceBootstrapper.configure_domain_specific_databases(container)
    
    # Validate configuration
    try:
        # Test core service resolution
        config_service = container.resolve(ConfigurationService)
        logger.debug("Successfully resolved ConfigurationService")
        
        # Test database service resolution
        connection_pool = container.resolve(IConnectionPool)
        logger.debug("Successfully resolved IConnectionPool")
        
        # Test logging service resolution
        trading_logger = container.resolve(TradingLogger)
        logger.debug("Successfully resolved TradingLogger")
        
    except Exception as e:
        logger.error(f"Failed to validate container configuration: {e}")
        raise
    
    logger.info("Application bootstrap completed successfully")
    return container


# Convenience decorators for common service patterns
def database_service(domain: str = 'technical_analysis'):
    """
    Decorator for database-dependent services.
    
    Args:
        domain: Database domain to use
    """
    def decorator(cls):
        # Auto-register with dependency on specific domain connection pool
        pool_type_name = f"{domain.title().replace('_', '')}ConnectionPool"
        # This would require runtime type creation, simplified for now
        return singleton()(cls)
    return decorator


def data_service():
    """Decorator for data collection services."""
    def decorator(cls):
        return singleton()(cls)
    return decorator


def trading_service():
    """Decorator for trading-related services.""" 
    def decorator(cls):
        return transient()(cls)
    return decorator


def analysis_service():
    """Decorator for technical analysis services."""
    def decorator(cls):
        return transient()(cls)
    return decorator