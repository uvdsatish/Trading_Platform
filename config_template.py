# Configuration Template
# Copy this file to config.py and fill in your actual values

# Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "your_database_name",
    "user": "your_username", 
    "password": "your_password",
    "port": "5432"
}

# IQFeed Configuration
IQFEED_CONFIG = {
    "host": "127.0.0.1",
    "port": 9100,
    "product_id": "YOUR_PRODUCT_ID",
    "login": "YOUR_LOGIN",
    "password": "YOUR_PASSWORD"
}

# Trading Parameters
TRADING_CONFIG = {
    "chunk_size": 500,  # Number of tickers to process at once
    "retry_attempts": 5,
    "retry_delay": 10,  # seconds
    "default_start_date": "19500101"
}