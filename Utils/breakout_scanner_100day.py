import psycopg2
import pandas as pd

def check_100day_breakouts(tickers):
    """
    Check if tickers have crossed above 100-day high or below 100-day low
    
    Args:
        tickers: List of ticker symbols to check
    
    Returns:
        DataFrame with breakout information
    """
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
        
        # Query to calculate 100-day high/low and check breakouts
        query = f"""
        WITH daily_data AS (
            SELECT ticker, date, close, high, low,
                   MAX(high) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 99 PRECEDING AND CURRENT ROW) as d100high,
                   MIN(low) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 99 PRECEDING AND CURRENT ROW) as d100low,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
            FROM key_indicators_alltickers 
            WHERE ticker IN ('{ticker_list}')
        )
        SELECT ticker, date, high, low, d100high, d100low,
               CASE 
                   WHEN high >= d100high THEN 'Above 100D High'
                   WHEN low <= d100low THEN 'Below 100D Low'
                   ELSE 'Within Range'
               END as breakout_status,
               ROUND(((high - d100high) / d100high * 100)::numeric, 2) as pct_above_high,
               ROUND(((low - d100low) / d100low * 100)::numeric, 2) as pct_above_low
        FROM daily_data 
        WHERE rn = 1 AND d100high IS NOT NULL
        ORDER BY ticker
        """
        
        # Execute query and get results
        df = pd.read_sql_query(query, connection)
        connection.close()
        
        # Filter only breakouts
        breakouts = df[df['breakout_status'] != 'Within Range']
        
        print(f"Checked {len(tickers)} tickers")
        print(f"Found {len(breakouts)} 100-day breakouts:")
        print("-" * 80)
        
        for _, row in breakouts.iterrows():
            if row['breakout_status'] == 'Above 100D High':
                print(f"{row['ticker']}: ${row['high']:.2f} - NEW 100D HIGH (+{row['pct_above_high']:.2f}%)")
            else:
                print(f"{row['ticker']}: ${row['low']:.2f} - NEW 100D LOW ({row['pct_above_low']:.2f}%)")
        
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Example ticker list
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN']
    
    results = check_100day_breakouts(test_tickers)
    
    if results is not None:
        print(f"\nFull results saved. Total tickers analyzed: {len(results)}")