import psycopg2
import pandas as pd

def check_52week_breakouts(tickers):
    """
    Check if tickers have crossed above 52-week high or below 52-week low
    
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
        
        # Query to get latest data for each ticker
        query = f"""
        WITH latest_data AS (
            SELECT ticker, date, high, low, w52high, w52low,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
            FROM key_indicators_alltickers 
            WHERE ticker IN ('{ticker_list}')
        )
        SELECT ticker, date, high, w52high, w52low,
               CASE 
                   WHEN high >= w52high THEN 'Above 52W High'
                   WHEN low <= w52low THEN 'Below 52W Low'
                   ELSE 'Within Range'
               END as breakout_status,
               ROUND(((high - w52high) / w52high * 100)::numeric, 2) as pct_above_high,
               ROUND(((low - w52low) / w52low * 100)::numeric, 2) as pct_above_low
        FROM latest_data 
        WHERE rn = 1
        ORDER BY ticker
        """
        
        # Execute query and get results
        df = pd.read_sql_query(query, connection)
        connection.close()
        
        # Filter only breakouts
        breakouts = df[df['breakout_status'] != 'Within Range']
        
        print(f"Checked {len(tickers)} tickers")
        print(f"Found {len(breakouts)} breakouts:")
        print("-" * 80)
        
        for _, row in breakouts.iterrows():
            if row['breakout_status'] == 'Above 52W High':
                print(f"{row['ticker']}: ${row['high']:.2f} - NEW 52W HIGH (+{row['pct_above_high']:.2f}%)")
            else:
                print(f"{row['ticker']}: ${row['low']:.2f} - NEW 52W LOW ({row['pct_above_low']:.2f}%)")
        
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Example ticker list
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN']
    
    results = check_52week_breakouts(test_tickers)
    
    if results is not None:
        print(f"\nFull results saved. Total tickers analyzed: {len(results)}")