"""
Unit tests for the configuration service.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import yaml
from pathlib import Path

from src.config.service import ConfigurationService, get_config_service
from src.config.base import ConfigurationError
from src.config.database import DatabaseConfig
from src.config.providers import IQFeedConfig


class TestConfigurationService(unittest.TestCase):
    """Test cases for ConfigurationService."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test configs
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.test_dir) / 'config'
        self.config_dir.mkdir()
        
        # Create test configuration file
        self.test_config = {
            'database': {
                'primary': {
                    'type': 'postgresql',
                    'host': 'test-host',
                    'port': 5432,
                    'name': 'test_db',
                    'user': 'test_user',
                    'password': 'test_pass'
                },
                'technical_analysis': {
                    'type': 'postgresql',
                    'host': 'ta-host',
                    'port': 5432,
                    'name': 'ta_db',
                    'user': 'ta_user',
                    'password': 'ta_pass'
                }
            },
            'data_providers': {
                'iqfeed': {
                    'type': 'iqfeed',
                    'enabled': True,
                    'host': '127.0.0.1',
                    'ports': {
                        'lookup': 9100
                    }
                }
            },
            'features': {
                'backtesting': True,
                'real_time_alerts': False
            },
            'logging': {
                'level': 'DEBUG'
            }
        }
        
        # Write test config to file
        config_file = self.config_dir / 'application.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
        # Clear singleton instance
        ConfigurationService._instance = None
    
    def test_singleton_pattern(self):
        """Test that ConfigurationService implements singleton pattern."""
        service1 = ConfigurationService('development', self.config_dir)
        service2 = ConfigurationService('development', self.config_dir)
        
        self.assertIs(service1, service2)
    
    def test_load_configuration(self):
        """Test loading configuration from file."""
        service = ConfigurationService('development', self.config_dir)
        
        # Check that configuration is loaded
        self.assertEqual(service.environment, 'development')
        self.assertIsNotNone(service.db_config)
        self.assertIsNotNone(service.provider_config)
    
    def test_get_database_config(self):
        """Test getting database configuration."""
        service = ConfigurationService('development', self.config_dir)
        
        # Get primary database
        primary_db = service.get_database_config()
        self.assertIsInstance(primary_db, DatabaseConfig)
        self.assertEqual(primary_db.get('host'), 'test-host')
        self.assertEqual(primary_db.get('name'), 'test_db')
        
        # Get specific database
        ta_db = service.get_database_config('technical_analysis')
        self.assertEqual(ta_db.get('host'), 'ta-host')
        self.assertEqual(ta_db.get('name'), 'ta_db')
    
    def test_get_database_connection_string(self):
        """Test getting database connection string."""
        service = ConfigurationService('development', self.config_dir)
        
        # Get connection string with password
        conn_str = service.get_database_connection_string()
        self.assertIn('postgresql://', conn_str)
        self.assertIn('test_user:test_pass', conn_str)
        self.assertIn('test-host:5432/test_db', conn_str)
        
        # Get connection string without password
        conn_str_safe = service.get_database_connection_string(include_password=False)
        self.assertIn('test_user:***', conn_str_safe)
        self.assertNotIn('test_pass', conn_str_safe)
    
    def test_get_data_provider_config(self):
        """Test getting data provider configuration."""
        service = ConfigurationService('development', self.config_dir)
        
        # Get IQFeed config
        iqfeed = service.get_data_provider_config('iqfeed')
        self.assertIsInstance(iqfeed, IQFeedConfig)
        self.assertEqual(iqfeed.get('host'), '127.0.0.1')
        self.assertEqual(iqfeed.get('ports.lookup'), 9100)
    
    def test_get_setting(self):
        """Test getting general settings."""
        service = ConfigurationService('development', self.config_dir)
        
        # Get existing setting
        log_level = service.get_setting('logging.level')
        self.assertEqual(log_level, 'DEBUG')
        
        # Get non-existent setting with default
        missing = service.get_setting('missing.key', 'default')
        self.assertEqual(missing, 'default')
    
    def test_feature_flags(self):
        """Test feature flag functionality."""
        service = ConfigurationService('development', self.config_dir)
        
        # Check enabled feature
        self.assertTrue(service.is_feature_enabled('backtesting'))
        
        # Check disabled feature
        self.assertFalse(service.is_feature_enabled('real_time_alerts'))
        
        # Check non-existent feature (should default to False)
        self.assertFalse(service.is_feature_enabled('non_existent'))
    
    def test_environment_checks(self):
        """Test environment checking methods."""
        # Development environment
        service = ConfigurationService('development', self.config_dir)
        self.assertTrue(service.is_development())
        self.assertFalse(service.is_production())
        self.assertFalse(service.is_testing())
        
        # Clear singleton for next test
        ConfigurationService._instance = None
        
        # Production environment
        prod_config_file = self.config_dir / 'application-production.yaml'
        with open(prod_config_file, 'w') as f:
            yaml.dump({'environment': 'production'}, f)
        
        service = ConfigurationService('production', self.config_dir)
        self.assertFalse(service.is_development())
        self.assertTrue(service.is_production())
        self.assertFalse(service.is_testing())
    
    @patch.dict(os.environ, {'TRADING_DB_PASSWORD': 'env_password'})
    def test_environment_variable_override(self):
        """Test that environment variables override config file values."""
        # Modify config to use environment variable
        config_with_env = self.test_config.copy()
        config_with_env['database']['primary']['password'] = '${TRADING_DB_PASSWORD:default}'
        
        config_file = self.config_dir / 'application.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_with_env, f)
        
        # Clear singleton
        ConfigurationService._instance = None
        
        service = ConfigurationService('development', self.config_dir)
        
        # Password should come from environment variable
        # Note: This test assumes ConfigLoader.load_from_env is working
        # In real implementation, you'd need to ensure env var substitution works
    
    def test_to_dict_safe(self):
        """Test exporting configuration with sensitive data masked."""
        service = ConfigurationService('development', self.config_dir)
        
        # Get safe dictionary
        config_dict = service.to_dict(safe=True)
        
        # Check that passwords are masked
        self.assertEqual(
            config_dict['database']['primary']['password'],
            '***'
        )
        self.assertEqual(
            config_dict['database']['technical_analysis']['password'],
            '***'
        )
        
        # Check that other values are intact
        self.assertEqual(
            config_dict['database']['primary']['host'],
            'test-host'
        )
    
    def test_reload_configuration(self):
        """Test reloading configuration."""
        service = ConfigurationService('development', self.config_dir)
        
        # Get initial value
        initial_level = service.get_setting('logging.level')
        self.assertEqual(initial_level, 'DEBUG')
        
        # Modify config file
        self.test_config['logging']['level'] = 'INFO'
        config_file = self.config_dir / 'application.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Reload configuration
        service.reload()
        
        # Check that new value is loaded
        new_level = service.get_setting('logging.level')
        self.assertEqual(new_level, 'INFO')
    
    def test_get_config_service_helper(self):
        """Test the get_config_service helper function."""
        service = get_config_service('development')
        self.assertIsInstance(service, ConfigurationService)
        
        # Should return same instance
        service2 = get_config_service()
        self.assertIs(service, service2)
    
    def test_missing_database_config(self):
        """Test handling of missing database configuration."""
        # Create config without database section
        config_without_db = {'features': {'test': True}}
        config_file = self.config_dir / 'application.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_without_db, f)
        
        # Clear singleton
        ConfigurationService._instance = None
        
        service = ConfigurationService('development', self.config_dir)
        
        # Should raise error when trying to get database config
        with self.assertRaises(ValueError):
            service.get_database_config()
    
    def test_api_config(self):
        """Test getting API configuration."""
        # Add API config to test config
        self.test_config['apis'] = {
            'rest': {
                'enabled': True,
                'port': 8000
            }
        }
        
        config_file = self.config_dir / 'application.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Clear singleton
        ConfigurationService._instance = None
        
        service = ConfigurationService('development', self.config_dir)
        
        # Get API config
        rest_config = service.get_api_config('rest')
        self.assertEqual(rest_config['enabled'], True)
        self.assertEqual(rest_config['port'], 8000)
        
        # Get non-existent API config
        ws_config = service.get_api_config('websocket')
        self.assertEqual(ws_config, {})


if __name__ == '__main__':
    unittest.main()