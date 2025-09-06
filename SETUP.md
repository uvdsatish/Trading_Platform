# Setup Guide

## Prerequisites

1. **Python 3.7+** installed
2. **PostgreSQL** database server
3. **IQFeed subscription** (for live market data)

## Installation Steps

### 1. Clone and Install
```bash
git clone https://github.com/yourusername/Trading_Platform.git
cd Trading_Platform
pip install -r requirements.txt
```

### 2. Database Setup
```sql
-- Create databases
CREATE DATABASE markets_technicals;
CREATE DATABASE markets_internals;
CREATE DATABASE Plurality;

-- Create required tables (examples)
-- Add your specific table schemas here
```

### 3. Configuration
```bash
# Copy configuration template
cp config_template.py config.py

# Edit config.py with your credentials
# - Database connection details
# - IQFeed login credentials
```

### 4. IQFeed Setup
- Install IQFeed client software
- Configure credentials in `localconfig/passwords.py`:
```python
dtn_product_id = "YOUR_PRODUCT_ID"
dtn_login = "YOUR_LOGIN"
dtn_password = "YOUR_PASSWORD"
```

### 5. Initial Data Load
```bash
# Download historical data
python Data_Management/Combined_iqfeed_upload.py

# Update ticker lists
python Data_Management/tickers_update.py
```

## Verification

Test your setup:
```bash
# Test database connection
python -c "import psycopg2; print('Database OK')"

# Test basic functionality
python Utils/breakout_scanner.py
```

## Troubleshooting

- **Database connection issues**: Check PostgreSQL service and credentials
- **IQFeed connection**: Ensure IQFeed client is running and credentials are correct
- **Missing data**: Run initial data download scripts
- **Permission errors**: Check file permissions and database user privileges