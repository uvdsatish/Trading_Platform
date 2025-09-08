# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

This is a Python-based trading platform. Key commands:

```bash
# Install dependencies
pip install -r requirements.txt

# Run main data collection scripts
python Data_Management/Combined_iqfeed_upload.py      # Historical data download
python Data_Management/IQDelta.py                     # Incremental data updates  
python Data_Management/tickers_update.py              # Update ticker lists

# Run core trading strategies
python Technicals/RunAway.py                          # Current momentum strategy
python Technicals/RunAway-historical.py              # Historical backtesting
python Utils/breakout_scanner.py                      # 52-week breakout scanner
python Utils/breakout_scanner_25day.py               # 25-day breakout scanner
python Utils/breakout_scanner_100day.py              # 100-day breakout scanner

# Run technical analysis
python Technicals/Plurality-WAMRS/Plurality-RS-upload.py        # Relative strength analysis
python Technicals/Key_Indicators_population/KeyIndicatorsPopulation.py         # Technical indicators
python Technicals/Key_Indicators_population/KeyIndicatorsPopulation_Delta_Spark.py  # Spark-based indicators
python Technicals/Daily_Tech_Criteria.py             # Daily screening criteria

# Run market internals analysis
python Internals/Hindenburg_Omen/HindenburgOmen_model.py       # Market crash indicator
python Internals/Hindenburg_Omen/data_loading.py              # Load internal market data

# Performance and trade analysis
python Technicals/Performance_analysis/Combined_performance_SPY.py      # SPY performance analysis
python Technicals/TradeLog_Pre.py                     # Pre-trade analysis
python Technicals/TradeLog_Post.py                    # Post-trade analysis
```

## Architecture Overview

### Core Modules
- **Data_Management/**: Market data acquisition via IQFeed, PostgreSQL storage, ticker management
- **Technicals/**: Technical analysis, trading strategies (RunAway, Plurality-WAMRS), breakout detection
- **Internals/**: Market breadth analysis, Hindenburg Omen crash indicator
- **Utils/**: Scanners, utility functions, data processing tools
- **Back_Testing/**: Performance analysis and backtesting frameworks

### Key Systems
- **Plurality-WAMRS**: Proprietary relative strength system for sector rotation analysis
- **RunAway Strategy**: Momentum-based trading system with historical backtesting capabilities  
- **IQFeed Integration**: Real-time and historical market data via pyiqfeed library
- **PostgreSQL Storage**: Multiple databases (markets_technicals, markets_internals, Plurality)
- **Spark Support**: Large-scale technical indicator calculations via PySpark

### Data Flow
1. **Data Ingestion**: Combined_iqfeed_upload.py → PostgreSQL (usstockseod table)
2. **Delta Updates**: IQDelta.py for incremental data updates
3. **Technical Analysis**: Key_Indicators_population/ modules → key_indicators tables
4. **Strategy Execution**: RunAway.py reads indicators → generates trade signals
5. **Performance Tracking**: TradeLog_Pre/Post.py for trade analysis

### Configuration
- Copy `config_template.py` to `config.py` with database and IQFeed credentials
- Set up `pyiqfeed/localconfig/passwords.py` and `Data_Management/pyiqfeed/localconfig/passwords.py`
- Configure PostgreSQL databases: markets_technicals, markets_internals, Plurality

### Database Schema Patterns
- **usstockseod**: Core OHLCV data with date, ticker, open, high, low, close, volume
- **key_indicators_***: Technical indicators with date, ticker, and calculated metrics
- **rs_industry_groups**: Relative strength by industry group and date
- Market internal tables follow *_composite_raw naming pattern

### Dependencies
Core: pandas, psycopg2-binary, numpy, pyspark
Analysis: matplotlib, seaborn, scipy
Data: openpyxl, xlrd, pyyaml
System: psutil for monitoring

### Development Notes
- All scripts expect PostgreSQL connection via config.py parameters
- IQFeed requires active subscription and running client
- Spark scripts (KeyIndicatorsPopulation_Delta_Spark.py) need PySpark configuration
- Date formats are typically YYYYMMDD strings for IQFeed compatibility
- Chunked processing (default 500 tickers) for large datasets