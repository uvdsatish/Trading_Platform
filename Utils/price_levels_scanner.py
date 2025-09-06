import psycopg2
import pandas as pd

def get_price_levels(tickers, direction='long'):
    """
    Get price levels for tickers based on direction
    
    Args:
        tickers: Single ticker string or list of tickers
        direction: 'long' (shows highs) or 'short' (shows lows)
    
    Returns:
        DataFrame with price levels
    """
    # Handle single ticker input
    if isinstance(tickers, str):
        tickers = [tickers]
    
    try:
        # Connect to database
        connection = psycopg2.connect(
            database="markets_technicals",
            user="postgres",
            password="root",
            host="localhost",
            port="5432"
        )
        
        # Create ticker list for SQL query
        ticker_list = "','".join(tickers)
        
        # Query to get price levels
        query = f"""
        WITH daily_data AS (
            SELECT ticker, date, close, high, low, w52high, w52low,
                   MAX(high) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 24 PRECEDING AND CURRENT ROW) as d25high,
                   MIN(low) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 24 PRECEDING AND CURRENT ROW) as d25low,
                   MAX(high) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 99 PRECEDING AND CURRENT ROW) as d100high,
                   MIN(low) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 99 PRECEDING AND CURRENT ROW) as d100low,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
            FROM key_indicators_alltickers 
            WHERE ticker IN ('{ticker_list}')
        )
        SELECT ticker, date, close, d25high, d25low, d100high, d100low, w52high, w52low
        FROM daily_data 
        WHERE rn = 1
        ORDER BY ticker
        """
        
        # Execute query and get results
        df = pd.read_sql_query(query, connection)
        connection.close()
        
        # Display results based on direction
        print(f"Price Levels for {direction.upper()} positions:")
        print("=" * 80)
        
        if direction.lower() == 'long':
            for _, row in df.iterrows():
                print(f"{row['ticker']} ({row['date']}):")
                print(f"  Current Close: ${row['close']:.2f}")
                print(f"  25-Day High:   ${row['d25high']:.2f}")
                print(f"  100-Day High:  ${row['d100high']:.2f}")
                print(f"  52-Week High:  ${row['w52high']:.2f}")
                print("-" * 40)
        else:  # short
            for _, row in df.iterrows():
                print(f"{row['ticker']} ({row['date']}):")
                print(f"  Current Close: ${row['close']:.2f}")
                print(f"  25-Day Low:    ${row['d25low']:.2f}")
                print(f"  100-Day Low:   ${row['d100low']:.2f}")
                print(f"  52-Week Low:   ${row['w52low']:.2f}")
                print("-" * 40)
        
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Single ticker
    print("LONG positions:")
    results_long = get_price_levels('AAPL', 'long')
    
    print("\nSHORT positions:")
    results_short = get_price_levels(['TSLA', 'NVDA'], 'short')