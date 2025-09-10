"""
Unit tests for the logging infrastructure.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import json
import logging
import tempfile
import os
from pathlib import Path
from datetime import datetime
import threading
import time

from src.infrastructure.logging.base import (
    TradingLogger,
    LogContext,
    LoggerFactory,
    generate_request_id,
    set_request_context,
    request_id_var,
    user_id_var,
    session_id_var
)
from src.infrastructure.logging.formatters import (
    JSONFormatter,
    ColoredConsoleFormatter,
    SimpleFormatter,
    AuditFormatter
)
from src.infrastructure.logging.handlers import (
    RotatingJSONFileHandler,
    MetricsHandler
)
from src.infrastructure.logging.performance import (
    PerformanceTimer,
    log_performance,
    OperationTracker,
    BatchOperationTimer
)


class TestTradingLogger(unittest.TestCase):
    """Test cases for TradingLogger."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger_name = 'test.logger'
        self.logger = TradingLogger(self.logger_name)
        
        # Mock the underlying Python logger
        self.mock_python_logger = Mock()
        self.logger.logger = self.mock_python_logger
    
    def test_singleton_behavior(self):
        """Test that TradingLogger implements singleton pattern per name."""
        logger1 = TradingLogger('same.name')
        logger2 = TradingLogger('same.name')
        logger3 = TradingLogger('different.name')
        
        self.assertIs(logger1, logger2)
        self.assertIsNot(logger1, logger3)
    
    def test_context_inclusion(self):
        """Test that context is included in log messages."""
        with set_request_context(request_id="test-123", user_id="user-456"):
            self.logger.info("Test message")
        
        # Check that the log was called with JSON containing context
        self.mock_python_logger.info.assert_called_once()
        call_args = self.mock_python_logger.info.call_args[0][0]
        
        log_data = json.loads(call_args)
        self.assertEqual(log_data['message'], "Test message")
        self.assertEqual(log_data['context']['request_id'], "test-123")
        self.assertEqual(log_data['context']['user_id'], "user-456")
    
    def test_error_with_exception(self):
        """Test error logging with exception details."""
        try:
            raise ValueError("Test exception")
        except Exception as e:
            self.logger.error("Error occurred", exception=e)
        
        self.mock_python_logger.error.assert_called_once()
        call_args = self.mock_python_logger.error.call_args[0][0]
        
        log_data = json.loads(call_args)
        self.assertEqual(log_data['message'], "Error occurred")
        self.assertIn('exception', log_data['extra'])
        self.assertEqual(log_data['extra']['exception']['type'], 'ValueError')
        self.assertEqual(log_data['extra']['exception']['message'], 'Test exception')
        self.assertIn('traceback', log_data['extra']['exception'])
    
    def test_log_performance(self):
        """Test performance logging."""
        self.logger.log_performance("test_operation", 1.5, records=100)
        
        self.mock_python_logger.info.assert_called_once()
        call_args = self.mock_python_logger.info.call_args[0][0]
        
        log_data = json.loads(call_args)
        self.assertIn("Performance:", log_data['message'])
        self.assertIn('performance', log_data['extra'])
        
        perf_data = log_data['extra']['performance']
        self.assertEqual(perf_data['operation'], 'test_operation')
        self.assertEqual(perf_data['duration_seconds'], 1.5)
        self.assertEqual(perf_data['duration_ms'], 1500)
        self.assertEqual(perf_data['records'], 100)
    
    def test_log_trade(self):
        """Test trade logging."""
        self.logger.log_trade("BUY", "AAPL", 100, 150.25, strategy="momentum")
        
        self.mock_python_logger.info.assert_called_once()
        call_args = self.mock_python_logger.info.call_args[0][0]
        
        log_data = json.loads(call_args)
        self.assertIn("Trade:", log_data['message'])
        self.assertIn('trade', log_data['extra'])
        
        trade_data = log_data['extra']['trade']
        self.assertEqual(trade_data['action'], 'BUY')
        self.assertEqual(trade_data['ticker'], 'AAPL')
        self.assertEqual(trade_data['quantity'], 100)
        self.assertEqual(trade_data['price'], 150.25)
        self.assertEqual(trade_data['total_value'], 15025.0)
        self.assertEqual(trade_data['strategy'], 'momentum')
    
    def test_log_data_fetch(self):
        """Test data fetching logging."""
        self.logger.log_data_fetch("IQFeed", "AAPL", 1000, 2.5, errors=2)
        
        self.mock_python_logger.info.assert_called_once()
        call_args = self.mock_python_logger.info.call_args[0][0]
        
        log_data = json.loads(call_args)
        self.assertIn("Data fetch:", log_data['message'])
        self.assertIn('data_fetch', log_data['extra'])
        
        fetch_data = log_data['extra']['data_fetch']
        self.assertEqual(fetch_data['source'], 'IQFeed')
        self.assertEqual(fetch_data['ticker'], 'AAPL')
        self.assertEqual(fetch_data['records'], 1000)
        self.assertEqual(fetch_data['duration_seconds'], 2.5)
        self.assertEqual(fetch_data['records_per_second'], 400.0)
        self.assertEqual(fetch_data['errors'], 2)


class TestLogContext(unittest.TestCase):
    """Test cases for LogContext."""
    
    def test_context_manager(self):
        """Test context manager functionality."""
        # Initially no context
        self.assertIsNone(request_id_var.get())
        self.assertIsNone(user_id_var.get())
        
        with LogContext(request_id="test-123", user_id="user-456"):
            # Context should be set
            self.assertEqual(request_id_var.get(), "test-123")
            self.assertEqual(user_id_var.get(), "user-456")
        
        # Context should be reset
        self.assertIsNone(request_id_var.get())
        self.assertIsNone(user_id_var.get())
    
    def test_nested_context(self):
        """Test nested context behavior."""
        with LogContext(request_id="outer", user_id="user1"):
            self.assertEqual(request_id_var.get(), "outer")
            self.assertEqual(user_id_var.get(), "user1")
            
            with LogContext(request_id="inner", session_id="session1"):
                self.assertEqual(request_id_var.get(), "inner")
                self.assertEqual(user_id_var.get(), "user1")  # Should persist
                self.assertEqual(session_id_var.get(), "session1")
            
            # Should revert to outer context
            self.assertEqual(request_id_var.get(), "outer")
            self.assertEqual(user_id_var.get(), "user1")
            self.assertIsNone(session_id_var.get())


class TestLoggerFactory(unittest.TestCase):
    """Test cases for LoggerFactory."""
    
    @patch('src.config.get_config_service')
    def test_get_logger(self, mock_get_config):
        """Test logger creation through factory."""
        mock_config = Mock()
        mock_config.get_setting.return_value = 'DEBUG'
        mock_get_config.return_value = mock_config
        
        logger = LoggerFactory.get_logger('test.factory')
        
        self.assertIsInstance(logger, TradingLogger)
        self.assertEqual(logger.name, 'test.factory')
    
    def test_domain_loggers(self):
        """Test domain-specific logger methods."""
        data_logger = LoggerFactory.get_data_logger()
        self.assertEqual(data_logger.name, 'trading.data_collection')
        
        tech_logger = LoggerFactory.get_technical_logger()
        self.assertEqual(tech_logger.name, 'trading.technical_analysis')
        
        internals_logger = LoggerFactory.get_internals_logger()
        self.assertEqual(internals_logger.name, 'trading.market_internals')
        
        trading_logger = LoggerFactory.get_trading_logger()
        self.assertEqual(trading_logger.name, 'trading.execution')
        
        backtest_logger = LoggerFactory.get_backtesting_logger()
        self.assertEqual(backtest_logger.name, 'trading.backtesting')


class TestFormatters(unittest.TestCase):
    """Test cases for log formatters."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.record = logging.LogRecord(
            name='test.logger',
            level=logging.INFO,
            pathname='/test/path.py',
            lineno=100,
            msg='Test message',
            args=(),
            exc_info=None
        )
    
    def test_json_formatter(self):
        """Test JSON formatter."""
        formatter = JSONFormatter()
        
        # Test with structured message
        structured_msg = {
            'message': 'Test message',
            'context': {'request_id': 'test-123'},
            'extra': {'key': 'value'}
        }
        self.record.msg = json.dumps(structured_msg)
        
        result = formatter.format(self.record)
        parsed = json.loads(result)
        
        self.assertEqual(parsed['message'], 'Test message')
        self.assertEqual(parsed['context']['request_id'], 'test-123')
        self.assertEqual(parsed['extra']['key'], 'value')
    
    def test_json_formatter_with_exception(self):
        """Test JSON formatter with exception info."""
        formatter = JSONFormatter(include_traceback=True)
        
        try:
            raise ValueError("Test exception")
        except Exception:
            self.record.exc_info = Exception.__traceback__
        
        result = formatter.format(self.record)
        parsed = json.loads(result)
        
        self.assertIn('exception', parsed)
    
    def test_colored_console_formatter(self):
        """Test colored console formatter."""
        formatter = ColoredConsoleFormatter(include_context=True)
        
        # Test with structured message
        structured_msg = {
            'message': 'Test message',
            'context': {'request_id': 'test-123'},
            'extra': {'trade': {'action': 'BUY', 'ticker': 'AAPL', 'quantity': 100, 'price': 150.0}}
        }
        self.record.msg = json.dumps(structured_msg)
        self.record.levelname = 'INFO'
        
        result = formatter.format(self.record)
        
        self.assertIn('Test message', result)
        self.assertIn('[test-123', result)  # Context ID
        self.assertIn('🔄', result)  # Trade emoji
        self.assertIn('BUY 100 AAPL @ $150.0', result)  # Trade details
    
    def test_simple_formatter(self):
        """Test simple formatter."""
        formatter = SimpleFormatter()
        
        # Test with structured message
        structured_msg = {
            'message': 'Test message',
            'context': {'request_id': 'test-123'}
        }
        self.record.msg = json.dumps(structured_msg)
        
        result = formatter.format(self.record)
        
        self.assertIn('Test message', result)
        self.assertIn('[test-123', result)  # Context ID should be included
    
    def test_audit_formatter(self):
        """Test audit formatter."""
        formatter = AuditFormatter()
        
        result = formatter.format(self.record)
        parsed = json.loads(result)
        
        # Check required audit fields
        self.assertIn('audit_timestamp', parsed)
        self.assertIn('record_timestamp', parsed)
        self.assertIn('integrity_hash', parsed)
        self.assertIn('process_id', parsed)
        self.assertIn('thread_id', parsed)


class TestPerformanceLogging(unittest.TestCase):
    """Test cases for performance logging."""
    
    def test_performance_timer(self):
        """Test PerformanceTimer context manager."""
        with patch('src.infrastructure.logging.base.LoggerFactory') as mock_factory:
            mock_logger = Mock()
            mock_factory.get_logger.return_value = mock_logger
            
            with PerformanceTimer('test_operation'):
                time.sleep(0.01)  # Small delay for timing
            
            # Check that performance was logged
            mock_logger.log_performance.assert_called_once()
            args = mock_logger.log_performance.call_args
            
            self.assertEqual(args[0][0], 'test_operation')  # operation name
            self.assertGreater(args[0][1], 0.005)  # duration should be > 5ms
    
    @patch('time.sleep', return_value=None)  # Speed up test
    def test_log_performance_decorator(self, mock_sleep):
        """Test @log_performance decorator."""
        with patch('src.infrastructure.logging.performance.PerformanceTimer') as mock_timer_class:
            mock_timer = Mock()
            mock_timer_class.return_value = mock_timer
            
            @log_performance('test_function')
            def test_function(x, y):
                return x + y
            
            result = test_function(1, 2)
            
            self.assertEqual(result, 3)
            mock_timer_class.assert_called_once_with(
                'test_function',
                include_memory=False,
                include_cpu=False
            )
            mock_timer.__enter__.assert_called_once()
            mock_timer.__exit__.assert_called_once()
    
    def test_operation_tracker(self):
        """Test OperationTracker."""
        tracker = OperationTracker('test_tracker')
        
        # Track some operations
        with tracker.track('operation_1'):
            time.sleep(0.01)
        
        with tracker.track('operation_1'):
            time.sleep(0.01)
        
        with tracker.track('operation_2'):
            time.sleep(0.01)
        
        # Check statistics
        stats = tracker.get_statistics()
        
        self.assertIn('operation_1', stats)
        self.assertIn('operation_2', stats)
        
        op1_stats = stats['operation_1']
        self.assertEqual(op1_stats['count'], 2)
        self.assertEqual(op1_stats['success_count'], 2)
        self.assertEqual(op1_stats['failure_count'], 0)
        self.assertGreater(op1_stats['avg_time'], 0)
        self.assertEqual(op1_stats['success_rate'], 1.0)
    
    def test_batch_operation_timer(self):
        """Test BatchOperationTimer."""
        with patch('src.infrastructure.logging.performance.LoggerFactory') as mock_factory:
            mock_logger = Mock()
            mock_factory.get_performance_logger.return_value = mock_logger
            
            with BatchOperationTimer('process_batch', 1000) as timer:
                timer.update(500)
                time.sleep(0.01)
            
            # Check that performance was logged with batch metrics
            mock_logger.log_performance.assert_called_once()
            args = mock_logger.log_performance.call_args
            
            self.assertEqual(args[0][0], 'process_batch')  # operation name
            self.assertIn('batch_size', args[1])
            self.assertIn('throughput_per_second', args[1])
            self.assertEqual(args[1]['batch_size'], 1000)
            self.assertEqual(args[1]['items_processed'], 500)


class TestMetricsHandler(unittest.TestCase):
    """Test cases for MetricsHandler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.metrics_file = os.path.join(self.temp_dir, 'test_metrics.json')
        self.handler = MetricsHandler(self.metrics_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_error_counting(self):
        """Test error counting functionality."""
        # Create error records
        error_record = logging.LogRecord(
            name='test.logger',
            level=logging.ERROR,
            pathname='/test/path.py',
            lineno=100,
            msg='Test error',
            args=(),
            exc_info=None
        )
        
        warning_record = logging.LogRecord(
            name='test.logger',
            level=logging.WARNING,
            pathname='/test/path.py',
            lineno=100,
            msg='Test warning',
            args=(),
            exc_info=None
        )
        
        # Emit records
        self.handler.emit(error_record)
        self.handler.emit(warning_record)
        
        # Check metrics
        metrics = self.handler.get_metrics()
        
        self.assertEqual(metrics['error_count'], 1)
        self.assertEqual(metrics['warning_count'], 1)
        self.assertIn('Test error', metrics['error_types'])
    
    def test_performance_metrics_extraction(self):
        """Test extraction of performance metrics from logs."""
        # Create performance log record
        perf_log = {
            'message': 'Performance test',
            'extra': {
                'performance': {
                    'operation': 'test_operation',
                    'duration_ms': 150.5
                }
            }
        }
        
        record = logging.LogRecord(
            name='test.logger',
            level=logging.INFO,
            pathname='/test/path.py',
            lineno=100,
            msg=json.dumps(perf_log),
            args=(),
            exc_info=None
        )
        
        # Emit record
        self.handler.emit(record)
        
        # Check metrics
        metrics = self.handler.get_metrics()
        
        self.assertEqual(len(metrics['performance_samples']), 1)
        sample = metrics['performance_samples'][0]
        self.assertEqual(sample['operation'], 'test_operation')
        self.assertEqual(sample['duration_ms'], 150.5)
    
    def test_trade_counting(self):
        """Test trade counting functionality."""
        # Create trade log record
        trade_log = {
            'message': 'Trade executed',
            'extra': {
                'trade': {
                    'action': 'BUY',
                    'ticker': 'AAPL',
                    'quantity': 100
                }
            }
        }
        
        record = logging.LogRecord(
            name='test.logger',
            level=logging.INFO,
            pathname='/test/path.py',
            lineno=100,
            msg=json.dumps(trade_log),
            args=(),
            exc_info=None
        )
        
        # Emit record
        self.handler.emit(record)
        
        # Check metrics
        metrics = self.handler.get_metrics()
        
        self.assertEqual(metrics['trade_count'], 1)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_generate_request_id(self):
        """Test request ID generation."""
        id1 = generate_request_id()
        id2 = generate_request_id()
        
        self.assertIsInstance(id1, str)
        self.assertIsInstance(id2, str)
        self.assertNotEqual(id1, id2)  # Should be unique
        self.assertEqual(len(id1), 36)  # UUID length
    
    def test_set_request_context(self):
        """Test set_request_context convenience function."""
        context = set_request_context(user_id="test_user", session_id="test_session")
        
        self.assertIsInstance(context, LogContext)
        
        with context:
            self.assertEqual(user_id_var.get(), "test_user")
            self.assertEqual(session_id_var.get(), "test_session")
            self.assertIsNotNone(request_id_var.get())  # Should auto-generate


if __name__ == '__main__':
    unittest.main()