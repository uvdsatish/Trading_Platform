# Database Connection Pool Migration Guide

This guide helps you migrate existing scripts from direct psycopg2 connections to the new OOP-based connection pool system.

## Overview of Changes

### Old Pattern (Direct psycopg2)
- Every script creates its own database connection
- Hardcoded credentials in each file
- No connection pooling or reuse
- No automatic retry on failures
- Manual connection management

### New Pattern (Connection Pool)
- Centralized connection management
- Configuration-based credentials
- Connection pooling and reuse
- Automatic retry with exponential backoff
- Automatic resource cleanup

## Migration Examples

### 1. Simple Query Migration

#### Before (IQDelta.py pattern):
```python
import psycopg2

# Hardcoded connection
conn = psycopg2.connect(
    database="markets_technicals",
    user="postgres",
    password="root",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

try:
    cursor.execute("SELECT * FROM usstockseod WHERE ticker = %s", (ticker,))
    results = cursor.fetchall()
finally:
    cursor.close()
    conn.close()
```

#### After (New Pattern):
```python
from src.infrastructure.database import get_technical_session

# Automatic connection management
with get_technical_session() as session:
    results = session.fetch_all(
        "SELECT * FROM usstockseod WHERE ticker = %s", 
        (ticker,)
    )
    # Connection automatically returned to pool
```

### 2. Transaction Migration

#### Before (Combined_iqfeed_upload.py pattern):
```python
import psycopg2

conn = psycopg2.connect(
    database="markets_technicals",
    user="postgres",
    password="root",
    host="localhost"
)
cursor = conn.cursor()

try:
    cursor.execute("BEGIN")
    cursor.execute("INSERT INTO usstockseod VALUES (%s, %s, %s)", data)
    cursor.execute("UPDATE metadata SET last_update = %s", (date,))
    conn.commit()
except Exception as e:
    conn.rollback()
    raise
finally:
    cursor.close()
    conn.close()
```

#### After (New Pattern):
```python
from src.infrastructure.database import DomainConnectionManager

# Automatic transaction management
with DomainConnectionManager.transaction('technical_analysis') as session:
    session.execute("INSERT INTO usstockseod VALUES (%s, %s, %s)", data)
    session.execute("UPDATE metadata SET last_update = %s", (date,))
    # Automatically commits on success, rolls back on exception
```

### 3. Batch Insert Migration

#### Before (KeyIndicatorsPopulation.py pattern):
```python
conn = psycopg2.connect(database="markets_technicals", user="postgres", password="root")
cursor = conn.cursor()

# Manual batch insert
data_to_insert = []
for row in large_dataset:
    data_to_insert.append((row['date'], row['ticker'], row['value']))
    
    if len(data_to_insert) >= 1000:
        cursor.executemany(
            "INSERT INTO key_indicators VALUES (%s, %s, %s)",
            data_to_insert
        )
        conn.commit()
        data_to_insert = []

# Insert remaining
if data_to_insert:
    cursor.executemany("INSERT INTO key_indicators VALUES (%s, %s, %s)", data_to_insert)
    conn.commit()

cursor.close()
conn.close()
```

#### After (New Pattern):
```python
from src.infrastructure.database import DomainConnectionManager
from src.infrastructure.database import BatchTransaction

# Automatic batch management
batch_tx = BatchTransaction(
    DomainConnectionManager.get_session_factory('technical_analysis').transaction_manager
)

with batch_tx.batch_insert('key_indicators', ['date', 'ticker', 'value']) as batch:
    for row in large_dataset:
        batch.add((row['date'], row['ticker'], row['value']))
    # Automatically flushes and commits
```

### 4. Multiple Database Access

#### Before (Accessing multiple databases):
```python
# Technical database
tech_conn = psycopg2.connect(database="markets_technicals", user="postgres", password="root")
tech_cursor = tech_conn.cursor()

# Internals database  
int_conn = psycopg2.connect(database="markets_internals", user="postgres", password="root")
int_cursor = int_conn.cursor()

# Plurality database
plur_conn = psycopg2.connect(database="Plurality", user="postgres", password="root")
plur_cursor = plur_conn.cursor()

# Use connections...
# Close all connections...
```

#### After (New Pattern):
```python
from src.infrastructure.database import (
    get_technical_session,
    get_internals_session,
    get_plurality_session
)

# Each domain has its own pool
with get_technical_session() as tech_session:
    tech_data = tech_session.fetch_all("SELECT * FROM indicators")

with get_internals_session() as int_session:
    internal_data = int_session.fetch_all("SELECT * FROM market_breadth")

with get_plurality_session() as plur_session:
    rs_data = plur_session.fetch_all("SELECT * FROM rs_industry_groups")
```

### 5. Error Handling with Retry

#### Before (No retry logic):
```python
import psycopg2
import time

def get_data_with_retry(ticker, max_retries=3):
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                database="markets_technicals",
                user="postgres",
                password="root"
            )
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM prices WHERE ticker = %s", (ticker,))
            result = cursor.fetchall()
            cursor.close()
            conn.close()
            return result
        except psycopg2.OperationalError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Manual exponential backoff
                continue
            raise
```

#### After (Automatic retry with exponential backoff):
```python
from src.infrastructure.database import get_technical_session
from src.infrastructure.database import with_retry, RetryPolicy

@with_retry(RetryPolicy(max_attempts=3))
def get_data(ticker):
    with get_technical_session() as session:
        return session.fetch_all(
            "SELECT * FROM prices WHERE ticker = %s",
            (ticker,)
        )

# Automatically retries with exponential backoff on connection failures
result = get_data('AAPL')
```

## Step-by-Step Migration Process

### Step 1: Update Configuration

1. Ensure `config/application.yaml` has your database settings:
```yaml
database:
  technical_analysis:
    type: postgresql
    host: localhost
    port: 5432
    name: markets_technicals
    user: postgres
    password: ${TRADING_DB_PASSWORD:root}
```

2. Set environment variables in `.env`:
```
TRADING_DB_PASSWORD=your_actual_password
```

### Step 2: Replace Imports

Replace psycopg2 imports with new infrastructure imports:

```python
# Old
import psycopg2
from psycopg2 import sql

# New
from src.infrastructure.database import get_technical_session
from src.infrastructure.database import DomainConnectionManager
```

### Step 3: Update Connection Code

Replace connection creation with session factory:

```python
# Old
conn = psycopg2.connect(...)

# New (choose appropriate domain)
with get_technical_session() as session:
    # Use session
```

### Step 4: Update Query Execution

Replace cursor operations with session methods:

```python
# Old
cursor.execute(query, params)
results = cursor.fetchall()

# New
results = session.fetch_all(query, params)
```

### Step 5: Remove Manual Cleanup

Remove all manual connection/cursor closing:

```python
# Old
try:
    # operations
finally:
    cursor.close()
    conn.close()

# New - cleanup is automatic
with get_session() as session:
    # operations
```

## Common Migration Patterns

### Pattern 1: Script with Main Function
```python
# Old
def main():
    conn = psycopg2.connect(...)
    try:
        # Script logic
    finally:
        conn.close()

# New
from src.infrastructure.database import get_technical_session

def main():
    with get_technical_session() as session:
        # Script logic
```

### Pattern 2: Class-Based Script
```python
# Old
class DataProcessor:
    def __init__(self):
        self.conn = psycopg2.connect(...)
    
    def process(self):
        cursor = self.conn.cursor()
        # Process data
        cursor.close()
    
    def __del__(self):
        self.conn.close()

# New
from src.infrastructure.database import DatabaseSessionFactory

class DataProcessor:
    def __init__(self):
        self.session_factory = DatabaseSessionFactory('technical_analysis')
    
    def process(self):
        with self.session_factory.get_session() as session:
            # Process data
```

### Pattern 3: Long-Running Script
```python
# Old
conn = psycopg2.connect(...)
while True:
    cursor = conn.cursor()
    # Process batch
    cursor.close()
    time.sleep(60)

# New
from src.infrastructure.database import get_technical_session

while True:
    with get_technical_session() as session:
        # Process batch - connection returned to pool
    time.sleep(60)
```

## Benefits After Migration

1. **Connection Pooling**: Reuse connections instead of creating new ones
2. **Automatic Retry**: Built-in exponential backoff for transient failures
3. **Resource Safety**: Guaranteed cleanup with context managers
4. **Configuration Management**: No more hardcoded credentials
5. **Better Performance**: Connection pooling reduces overhead
6. **Domain Separation**: Clear separation between different databases
7. **Monitoring**: Built-in metrics and health checks
8. **Thread Safety**: Safe for concurrent access

## Testing Your Migration

After migrating a script, test it with:

```python
# Check pool statistics
from src.infrastructure.database import DomainConnectionManager

stats = DomainConnectionManager.get_stats('technical_analysis')
print(f"Active connections: {stats['active_connections']}")
print(f"Idle connections: {stats['idle_connections']}")
print(f"Total connections created: {stats['connections_created']}")
```

## Troubleshooting

### Issue: Import Errors
```python
# Make sure to add to Python path
import sys
sys.path.append('/path/to/Trading_Platform')
```

### Issue: Connection Timeout
```python
# Increase timeout in config
database:
  technical_analysis:
    pool:
      timeout: 60  # Increase from default 30
```

### Issue: Pool Exhaustion
```python
# Increase pool size in config
database:
  technical_analysis:
    pool:
      size: 10  # Increase minimum
      max_overflow: 30  # Increase maximum
```

## Migration Checklist

For each script:
- [ ] Remove hardcoded credentials
- [ ] Replace psycopg2 imports
- [ ] Replace connection creation with session factory
- [ ] Update query execution to use session methods
- [ ] Remove manual connection/cursor closing
- [ ] Add proper error handling
- [ ] Test with sample data
- [ ] Monitor pool statistics
- [ ] Document any script-specific changes

## Example: Full Script Migration

### Original Script (IQDelta.py simplified)
```python
import psycopg2
from datetime import datetime

def update_ticker_data(ticker, data):
    conn = psycopg2.connect(
        database="markets_technicals",
        user="postgres",
        password="root",
        host="localhost"
    )
    cursor = conn.cursor()
    
    try:
        # Check last update
        cursor.execute(
            "SELECT MAX(date) FROM usstockseod WHERE ticker = %s",
            (ticker,)
        )
        last_date = cursor.fetchone()[0]
        
        # Insert new data
        for row in data:
            if row['date'] > last_date:
                cursor.execute(
                    "INSERT INTO usstockseod VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (row['date'], ticker, row['open'], row['high'], 
                     row['low'], row['close'], row['volume'])
                )
        
        conn.commit()
        print(f"Updated {ticker}")
        
    except Exception as e:
        conn.rollback()
        print(f"Error updating {ticker}: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    for ticker in tickers:
        data = fetch_data(ticker)  # Assume this gets data
        update_ticker_data(ticker, data)
```

### Migrated Script (Using New System)
```python
from src.infrastructure.database import DomainConnectionManager
from src.infrastructure.database import with_retry, RetryPolicy
from datetime import datetime

@with_retry(RetryPolicy(max_attempts=3))
def update_ticker_data(ticker, data):
    with DomainConnectionManager.transaction('technical_analysis') as session:
        # Check last update
        result = session.fetch_one(
            "SELECT MAX(date) FROM usstockseod WHERE ticker = %s",
            (ticker,)
        )
        last_date = result[0] if result else None
        
        # Insert new data
        new_rows = 0
        for row in data:
            if not last_date or row['date'] > last_date:
                session.execute(
                    "INSERT INTO usstockseod VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (row['date'], ticker, row['open'], row['high'],
                     row['low'], row['close'], row['volume'])
                )
                new_rows += 1
        
        print(f"Updated {ticker}: {new_rows} new rows")
        # Transaction automatically commits

if __name__ == "__main__":
    # Initialize pools at startup
    from src.infrastructure.database import initialize_pools
    initialize_pools('technical_analysis')
    
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    for ticker in tickers:
        data = fetch_data(ticker)  # Assume this gets data
        update_ticker_data(ticker, data)
    
    # Check pool statistics
    from src.infrastructure.database import DomainConnectionManager
    stats = DomainConnectionManager.get_stats('technical_analysis')
    print(f"Pool stats: {stats}")
```

## Next Steps

1. Start with non-critical scripts for practice
2. Migrate data collection scripts (IQFeed related)
3. Migrate analysis scripts (Technical indicators)
4. Migrate reporting/output scripts
5. Update cron jobs/schedulers to use new pattern
6. Monitor pool metrics in production

## Support

For questions or issues during migration:
1. Check pool statistics for connection issues
2. Enable debug logging to trace connection lifecycle
3. Review the unit tests for usage examples
4. Document any script-specific challenges for team knowledge sharing