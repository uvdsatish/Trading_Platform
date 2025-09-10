"""
Transaction management with automatic commit/rollback and nested transaction support.
"""

from contextlib import contextmanager
from typing import Any, Optional, Generator, Dict
import logging
import uuid
import threading
from enum import Enum

from .exceptions import TransactionError, DeadlockError


class IsolationLevel(Enum):
    """PostgreSQL transaction isolation levels."""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


class TransactionManager:
    """
    Manages database transactions with support for nested transactions using savepoints.
    """
    
    def __init__(self, connection_manager):
        """
        Initialize transaction manager.
        
        Args:
            connection_manager: Connection manager instance
        """
        self.connection_manager = connection_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self._local = threading.local()
    
    @contextmanager
    def transaction(self, 
                   isolation_level: Optional[IsolationLevel] = None,
                   read_only: bool = False,
                   deferrable: bool = False) -> Generator[Any, None, None]:
        """
        Context manager for database transactions with automatic commit/rollback.
        
        Args:
            isolation_level: Transaction isolation level
            read_only: Whether transaction is read-only
            deferrable: Whether transaction is deferrable (for read-only transactions)
            
        Yields:
            Transaction context
            
        Example:
            with transaction_manager.transaction() as tx:
                cursor = tx.cursor()
                cursor.execute("INSERT INTO table VALUES (%s)", (value,))
                # Automatically commits on success, rolls back on exception
        """
        # Check if we're already in a transaction (nested transaction)
        if hasattr(self._local, 'transaction_stack') and self._local.transaction_stack:
            # Use savepoint for nested transaction
            yield from self._nested_transaction()
        else:
            # Start new top-level transaction
            yield from self._top_level_transaction(isolation_level, read_only, deferrable)
    
    @contextmanager
    def _top_level_transaction(self,
                              isolation_level: Optional[IsolationLevel],
                              read_only: bool,
                              deferrable: bool) -> Generator[Any, None, None]:
        """
        Manage top-level transaction.
        
        Args:
            isolation_level: Transaction isolation level
            read_only: Whether transaction is read-only
            deferrable: Whether transaction is deferrable
            
        Yields:
            Transaction context
        """
        # Initialize transaction stack
        if not hasattr(self._local, 'transaction_stack'):
            self._local.transaction_stack = []
        
        # Get connection
        with self.connection_manager.get_connection() as conn:
            tx_id = str(uuid.uuid4())[:8]
            
            try:
                # Set transaction properties
                if isolation_level:
                    self._set_isolation_level(conn, isolation_level)
                
                if read_only:
                    self._set_read_only(conn, True)
                
                if deferrable and read_only:
                    self._set_deferrable(conn, True)
                
                # Start transaction
                conn.autocommit = False
                
                # Create transaction context
                tx_context = TransactionContext(conn, tx_id, is_nested=False)
                self._local.transaction_stack.append(tx_context)
                
                self.logger.debug(f"Started transaction {tx_id}")
                
                yield tx_context
                
                # Commit if no exception
                conn.commit()
                self.logger.debug(f"Committed transaction {tx_id}")
                
            except Exception as e:
                # Rollback on exception
                try:
                    conn.rollback()
                    self.logger.debug(f"Rolled back transaction {tx_id}")
                except Exception as rollback_error:
                    self.logger.error(f"Error during rollback: {rollback_error}")
                
                # Check for deadlock
                if self._is_deadlock_error(e):
                    raise DeadlockError(f"Deadlock detected in transaction {tx_id}") from e
                
                raise TransactionError(f"Transaction {tx_id} failed: {e}") from e
            
            finally:
                # Clean up transaction stack
                if self._local.transaction_stack:
                    self._local.transaction_stack.pop()
                
                # Reset connection properties
                conn.autocommit = False
    
    @contextmanager
    def _nested_transaction(self) -> Generator[Any, None, None]:
        """
        Manage nested transaction using savepoints.
        
        Yields:
            Transaction context
        """
        # Get parent transaction
        parent_tx = self._local.transaction_stack[-1]
        conn = parent_tx.connection
        
        # Generate savepoint name
        savepoint_name = f"sp_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create savepoint
            cursor = conn.cursor()
            cursor.execute(f"SAVEPOINT {savepoint_name}")
            cursor.close()
            
            # Create nested transaction context
            tx_context = TransactionContext(
                conn, 
                savepoint_name, 
                is_nested=True,
                parent=parent_tx
            )
            self._local.transaction_stack.append(tx_context)
            
            self.logger.debug(f"Created savepoint {savepoint_name}")
            
            yield tx_context
            
            # Release savepoint if no exception
            cursor = conn.cursor()
            cursor.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            cursor.close()
            
            self.logger.debug(f"Released savepoint {savepoint_name}")
            
        except Exception as e:
            # Rollback to savepoint
            try:
                cursor = conn.cursor()
                cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                cursor.close()
                self.logger.debug(f"Rolled back to savepoint {savepoint_name}")
            except Exception as rollback_error:
                self.logger.error(f"Error rolling back to savepoint: {rollback_error}")
            
            raise TransactionError(f"Nested transaction {savepoint_name} failed: {e}") from e
        
        finally:
            # Remove from transaction stack
            if self._local.transaction_stack:
                self._local.transaction_stack.pop()
    
    def _set_isolation_level(self, conn: Any, level: IsolationLevel) -> None:
        """
        Set transaction isolation level.
        
        Args:
            conn: Database connection
            level: Isolation level
        """
        cursor = conn.cursor()
        cursor.execute(f"SET TRANSACTION ISOLATION LEVEL {level.value}")
        cursor.close()
    
    def _set_read_only(self, conn: Any, read_only: bool) -> None:
        """
        Set transaction read-only mode.
        
        Args:
            conn: Database connection
            read_only: Whether transaction is read-only
        """
        cursor = conn.cursor()
        mode = "READ ONLY" if read_only else "READ WRITE"
        cursor.execute(f"SET TRANSACTION {mode}")
        cursor.close()
    
    def _set_deferrable(self, conn: Any, deferrable: bool) -> None:
        """
        Set transaction deferrable mode.
        
        Args:
            conn: Database connection
            deferrable: Whether transaction is deferrable
        """
        cursor = conn.cursor()
        mode = "DEFERRABLE" if deferrable else "NOT DEFERRABLE"
        cursor.execute(f"SET TRANSACTION {mode}")
        cursor.close()
    
    def _is_deadlock_error(self, error: Exception) -> bool:
        """
        Check if error is a deadlock error.
        
        Args:
            error: Exception to check
            
        Returns:
            True if deadlock error
        """
        error_message = str(error).lower()
        return 'deadlock' in error_message or 'lock timeout' in error_message
    
    @contextmanager
    def read_only_transaction(self) -> Generator[Any, None, None]:
        """
        Convenience method for read-only transactions.
        
        Yields:
            Transaction context
        """
        with self.transaction(
            isolation_level=IsolationLevel.READ_COMMITTED,
            read_only=True,
            deferrable=True
        ) as tx:
            yield tx
    
    @contextmanager
    def serializable_transaction(self) -> Generator[Any, None, None]:
        """
        Convenience method for serializable transactions.
        
        Yields:
            Transaction context
        """
        with self.transaction(isolation_level=IsolationLevel.SERIALIZABLE) as tx:
            yield tx


class TransactionContext:
    """
    Context object for active transactions.
    """
    
    def __init__(self, 
                 connection: Any,
                 transaction_id: str,
                 is_nested: bool = False,
                 parent: Optional['TransactionContext'] = None):
        """
        Initialize transaction context.
        
        Args:
            connection: Database connection
            transaction_id: Transaction or savepoint identifier
            is_nested: Whether this is a nested transaction
            parent: Parent transaction context (for nested transactions)
        """
        self.connection = connection
        self.transaction_id = transaction_id
        self.is_nested = is_nested
        self.parent = parent
        self._cursors = []
    
    def cursor(self, **kwargs) -> Any:
        """
        Create a cursor for this transaction.
        
        Args:
            **kwargs: Arguments to pass to cursor creation
            
        Returns:
            Database cursor
        """
        cursor = self.connection.cursor(**kwargs)
        self._cursors.append(cursor)
        return cursor
    
    def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """
        Execute a query in this transaction.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Query result
        """
        cursor = self.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor
        except Exception:
            cursor.close()
            raise
    
    def close_cursors(self) -> None:
        """Close all cursors created in this transaction."""
        for cursor in self._cursors:
            try:
                cursor.close()
            except:
                pass
        self._cursors.clear()


class BatchTransaction:
    """
    Manages batch operations within a transaction for improved performance.
    """
    
    def __init__(self, transaction_manager: TransactionManager, batch_size: int = 1000):
        """
        Initialize batch transaction.
        
        Args:
            transaction_manager: Transaction manager instance
            batch_size: Number of operations to batch
        """
        self.transaction_manager = transaction_manager
        self.batch_size = batch_size
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @contextmanager
    def batch_insert(self, table: str, columns: list) -> Generator[Any, None, None]:
        """
        Context manager for batch inserts.
        
        Args:
            table: Table name
            columns: Column names
            
        Yields:
            Batch inserter
            
        Example:
            with batch_transaction.batch_insert('prices', ['date', 'ticker', 'price']) as batch:
                for row in data:
                    batch.add(row)
                # Automatically flushes remaining rows
        """
        with self.transaction_manager.transaction() as tx:
            inserter = BatchInserter(tx, table, columns, self.batch_size)
            
            try:
                yield inserter
                inserter.flush()  # Flush any remaining rows
            except Exception as e:
                self.logger.error(f"Batch insert failed: {e}")
                raise


class BatchInserter:
    """
    Helper class for batch insert operations.
    """
    
    def __init__(self, tx_context: TransactionContext, 
                 table: str, columns: list, batch_size: int):
        """
        Initialize batch inserter.
        
        Args:
            tx_context: Transaction context
            table: Table name
            columns: Column names
            batch_size: Batch size
        """
        self.tx_context = tx_context
        self.table = table
        self.columns = columns
        self.batch_size = batch_size
        self.buffer = []
        self.count = 0
    
    def add(self, values: tuple) -> None:
        """
        Add values to batch.
        
        Args:
            values: Values to insert
        """
        self.buffer.append(values)
        
        if len(self.buffer) >= self.batch_size:
            self.flush()
    
    def flush(self) -> None:
        """Flush buffered rows to database."""
        if not self.buffer:
            return
        
        # Build insert query
        placeholders = ','.join(['%s'] * len(self.columns))
        query = f"INSERT INTO {self.table} ({','.join(self.columns)}) VALUES ({placeholders})"
        
        # Execute batch insert
        cursor = self.tx_context.cursor()
        cursor.executemany(query, self.buffer)
        cursor.close()
        
        self.count += len(self.buffer)
        self.buffer.clear()