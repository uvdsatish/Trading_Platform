# Trading Platform

A comprehensive Python-based algorithmic trading and market analysis platform for professional market research and trading strategy development.

## Features

### 📊 Data Management
- **IQFeed Integration**: Real-time and historical market data acquisition
- **PostgreSQL Storage**: Robust database management for market data
- **Automated Updates**: Scheduled data downloads and ticker management

### 📈 Technical Analysis
- **Plurality-WAMRS System**: Proprietary relative strength analysis
- **Industry Group Analysis**: Sector rotation and strength tracking
- **Breakout Detection**: Multi-timeframe breakout scanners
- **RunAway Strategy**: Momentum-based trading system with historical backtesting
- **Key Indicators Population**: Automated technical indicator calculations with Spark support
- **Daily Technical Criteria**: Screening system for daily trading opportunities

### 🔍 Market Internals
- **Hindenburg Omen**: Market crash prediction indicator
- **Market Breadth**: New highs/lows and advance/decline analysis
- **McClellan Oscillator**: Market momentum tracking

### 🎯 Trading Tools
- **Performance Analysis**: Comprehensive backtesting and trade evaluation
- **Price Level Scanner**: Support/resistance identification
- **Risk Management**: ATR-based stops and position sizing
- **Trade Logging**: Pre and post-trade analysis and metadata tracking
- **Drawdown/Runup Analysis**: Risk assessment and visualization tools
- **Price Movement Calculators**: Multiple versions for different analysis needs

## Project Structure

```
Trading_Platform/
├── Data_Management/     # Market data acquisition and storage
├── Technicals/         # Technical analysis and indicators
├── Internals/          # Market internals and breadth analysis
├── Back_Testing/       # Performance analysis and backtesting
├── Utils/              # Utility functions and scanners
├── Fundamentals/       # Fundamental analysis (future)
└── Macros/            # Macro economic analysis (future)
```

## Requirements

- Python 3.7+
- PostgreSQL database
- IQFeed subscription (for live data)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Trading_Platform.git
cd Trading_Platform
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up PostgreSQL database and configure connection parameters

4. Configure IQFeed credentials (see setup instructions)

## Key Components

### Data Management
- **Combined_iqfeed_upload.py**: Historical data downloader
- **IQDelta.py**: Incremental data updates
- **tickers_update.py**: Ticker list management

### Technical Analysis
- **Plurality-RS-upload.py**: Relative strength calculations
- **breakout_scanner.py**: Breakout detection system
- **RunAway.py**: Current momentum trading strategy
- **RunAway-historical.py**: Historical backtesting for RunAway strategy
- **Daily_Tech_Criteria.py**: Daily technical screening criteria
- **Key_Indicators_population/**: Technical indicator calculation modules with Spark support
- **Performance_analysis/**: SPY and ticker performance analysis tools
- **TradeLog_Pre.py / TradeLog_Post.py**: Trade analysis and logging
- **drawdown_runup_graph.py**: Risk visualization tools

### Market Internals
- **HindenburgOmen_model.py**: Market crash indicator
- **data_loading.py**: Internal market data processing

## Usage

### Basic Data Download
```python
# Download historical data for all tickers
python Data_Management/Combined_iqfeed_upload.py
```

### Run Breakout Scanner
```python
# Scan for 52-week breakouts
python Utils/breakout_scanner.py
```

### Run RunAway Strategy
```python
# Execute momentum trading strategy
python Technicals/RunAway.py
```

### Calculate Key Indicators
```python
# Populate technical indicators with Spark
python Technicals/Key_Indicators_population/KeyIndicatorsPopulation_Delta_Spark.py
```

### Calculate Hindenburg Omen
```python
# Generate market crash signals
python Internals/Hindenburg_Omen/HindenburgOmen_model.py
```

## Configuration

Create configuration files for:
- Database connection parameters
- IQFeed credentials
- Trading parameters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with your broker's and data provider's terms of service.

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

## Contact

For questions or support, please open an issue on GitHub.