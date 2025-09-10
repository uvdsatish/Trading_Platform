"""
Unit tests for retry logic and exponential backoff.
"""

import unittest
from unittest.mock import Mock, patch, call
import time
from psycopg2 import OperationalError, DatabaseError

from src.infrastructure.database.retry import (
    RetryPolicy,
    with_retry,
    RetryableOperation,
    CircuitBreaker
)
from src.infrastructure.database.exceptions import (
    RetryableError,
    DeadlockError,
    DatabaseTimeoutError
)


class TestRetryPolicy(unittest.TestCase):
    """Test cases for RetryPolicy."""
    
    def test_calculate_delay_exponential(self):
        """Test exponential backoff calculation."""
        policy = RetryPolicy(
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=False
        )
        
        # Test exponential growth
        self.assertEqual(policy.calculate_delay(0), 1.0)  # 1 * 2^0
        self.assertEqual(policy.calculate_delay(1), 2.0)  # 1 * 2^1
        self.assertEqual(policy.calculate_delay(2), 4.0)  # 1 * 2^2
        self.assertEqual(policy.calculate_delay(3), 8.0)  # 1 * 2^3
    
    def test_calculate_delay_with_max(self):
        """Test that delay is capped at max_delay."""
        policy = RetryPolicy(
            initial_delay=1.0,
            max_delay=5.0,
            exponential_base=2.0,
            jitter=False
        )
        
        # Should be capped at 5.0
        self.assertEqual(policy.calculate_delay(0), 1.0)
        self.assertEqual(policy.calculate_delay(1), 2.0)
        self.assertEqual(policy.calculate_delay(2), 4.0)
        self.assertEqual(policy.calculate_delay(3), 5.0)  # Capped
        self.assertEqual(policy.calculate_delay(10), 5.0)  # Still capped
    
    def test_calculate_delay_with_jitter(self):
        """Test that jitter adds randomness."""
        policy = RetryPolicy(
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=True
        )
        
        # With jitter, delays should vary
        delays = [policy.calculate_delay(1) for _ in range(10)]
        
        # All should be in range [1.0, 3.0] (2.0 * 0.5 to 2.0 * 1.5)
        for delay in delays:
            self.assertGreaterEqual(delay, 1.0)
            self.assertLessEqual(delay, 3.0)
        
        # Should have some variation
        self.assertGreater(len(set(delays)), 1)
    
    def test_should_retry_attempts(self):
        """Test retry decision based on attempts."""
        policy = RetryPolicy(max_attempts=3)
        error = RetryableError("Test error")
        
        self.assertTrue(policy.should_retry(error, 0))  # First retry
        self.assertTrue(policy.should_retry(error, 1))  # Second retry
        self.assertFalse(policy.should_retry(error, 2))  # No more retries
    
    def test_is_retryable_error(self):
        """Test identification of retryable errors."""
        policy = RetryPolicy()
        
        # Retryable errors
        self.assertTrue(policy.is_retryable_error(RetryableError("test")))
        self.assertTrue(policy.is_retryable_error(
            OperationalError("could not connect to server")
        ))
        self.assertTrue(policy.is_retryable_error(
            DatabaseError("deadlock detected")
        ))
        
        # Non-retryable errors
        self.assertFalse(policy.is_retryable_error(ValueError("test")))
        self.assertFalse(policy.is_retryable_error(TypeError("test")))


class TestWithRetryDecorator(unittest.TestCase):
    """Test cases for @with_retry decorator."""
    
    @patch('time.sleep')
    def test_successful_operation(self, mock_sleep):
        """Test that successful operations don't retry."""
        mock_func = Mock(return_value="success")
        
        @with_retry(RetryPolicy(max_attempts=3))
        def test_func():
            return mock_func()
        
        result = test_func()
        
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 1)
        mock_sleep.assert_not_called()
    
    @patch('time.sleep')
    def test_retry_on_failure(self, mock_sleep):
        """Test that failures trigger retries."""
        mock_func = Mock(side_effect=[
            RetryableError("fail"),
            RetryableError("fail"),
            "success"
        ])
        
        @with_retry(RetryPolicy(max_attempts=3, initial_delay=1.0, jitter=False))
        def test_func():
            return mock_func()
        
        result = test_func()
        
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 3)
        
        # Check delays (exponential backoff)
        expected_delays = [1.0, 2.0]  # After 1st and 2nd failures
        actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
        self.assertEqual(actual_delays, expected_delays)
    
    @patch('time.sleep')
    def test_max_attempts_exceeded(self, mock_sleep):
        """Test that max attempts are respected."""
        mock_func = Mock(side_effect=RetryableError("Always fails"))
        
        @with_retry(RetryPolicy(max_attempts=3))
        def test_func():
            return mock_func()
        
        with self.assertRaises(RetryableError):
            test_func()
        
        self.assertEqual(mock_func.call_count, 3)
    
    @patch('time.sleep')
    def test_non_retryable_error(self, mock_sleep):
        """Test that non-retryable errors don't trigger retries."""
        mock_func = Mock(side_effect=ValueError("Not retryable"))
        
        @with_retry(RetryPolicy(max_attempts=3))
        def test_func():
            return mock_func()
        
        with self.assertRaises(ValueError):
            test_func()
        
        self.assertEqual(mock_func.call_count, 1)
        mock_sleep.assert_not_called()


class TestRetryableOperation(unittest.TestCase):
    """Test cases for RetryableOperation context manager."""
    
    @patch('time.sleep')
    def test_successful_operation(self, mock_sleep):
        """Test successful operation without retries."""
        with RetryableOperation("test_op") as op:
            # Successful operation
            pass
        
        self.assertEqual(op.attempt, 0)
        mock_sleep.assert_not_called()
    
    @patch('time.sleep')
    def test_execute_with_retry(self, mock_sleep):
        """Test execute method with retries."""
        mock_func = Mock(side_effect=[
            RetryableError("fail"),
            "success"
        ])
        
        op = RetryableOperation(
            "test_op",
            RetryPolicy(max_attempts=3, initial_delay=1.0, jitter=False)
        )
        
        result = op.execute(mock_func)
        
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 2)


class TestCircuitBreaker(unittest.TestCase):
    """Test cases for CircuitBreaker."""
    
    def test_closed_state_success(self):
        """Test circuit breaker in closed state with successful calls."""
        breaker = CircuitBreaker(failure_threshold=3)
        mock_func = Mock(return_value="success")
        
        # Successful calls should work
        for _ in range(5):
            result = breaker.call(mock_func)
            self.assertEqual(result, "success")
        
        self.assertEqual(breaker.state, 'closed')
        self.assertEqual(breaker.failure_count, 0)
    
    def test_circuit_opens_on_failures(self):
        """Test that circuit opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3)
        mock_func = Mock(side_effect=Exception("fail"))
        
        # First failures should increment counter
        for i in range(3):
            with self.assertRaises(Exception):
                breaker.call(mock_func)
            
            if i < 2:
                self.assertEqual(breaker.state, 'closed')
            else:
                self.assertEqual(breaker.state, 'open')
        
        # Circuit should now be open
        self.assertEqual(breaker.state, 'open')
        self.assertEqual(breaker.failure_count, 3)
        
        # Further calls should be rejected
        with self.assertRaises(ConnectionError) as context:
            breaker.call(mock_func)
        
        self.assertIn("Circuit breaker is open", str(context.exception))
    
    @patch('time.time')
    def test_half_open_state(self, mock_time):
        """Test circuit breaker half-open state."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=60.0
        )
        mock_func = Mock()
        
        # Open the circuit
        mock_func.side_effect = Exception("fail")
        for _ in range(2):
            with self.assertRaises(Exception):
                breaker.call(mock_func)
        
        self.assertEqual(breaker.state, 'open')
        
        # Simulate time passing
        mock_time.return_value = time.time() + 61
        
        # Next call should try half-open
        mock_func.side_effect = None
        mock_func.return_value = "success"
        
        result = breaker.call(mock_func)
        
        # Should succeed and close circuit
        self.assertEqual(result, "success")
        self.assertEqual(breaker.state, 'closed')
        self.assertEqual(breaker.failure_count, 0)
    
    def test_manual_reset(self):
        """Test manual circuit breaker reset."""
        breaker = CircuitBreaker(failure_threshold=2)
        mock_func = Mock(side_effect=Exception("fail"))
        
        # Open the circuit
        for _ in range(2):
            with self.assertRaises(Exception):
                breaker.call(mock_func)
        
        self.assertEqual(breaker.state, 'open')
        
        # Manual reset
        breaker.reset()
        
        self.assertEqual(breaker.state, 'closed')
        self.assertEqual(breaker.failure_count, 0)
    
    def test_get_state(self):
        """Test getting circuit breaker state."""
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0
        )
        
        state = breaker.get_state()
        
        self.assertEqual(state['state'], 'closed')
        self.assertEqual(state['failure_count'], 0)
        self.assertEqual(state['failure_threshold'], 5)
        self.assertEqual(state['recovery_timeout'], 30.0)


if __name__ == '__main__':
    unittest.main()