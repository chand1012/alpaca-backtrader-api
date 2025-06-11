[![Build](https://github.com/chand1012/alpaca-backtrader-api/actions/workflows/build.yml/badge.svg)](https://github.com/chand1012/alpaca-backtrader-api/actions/workflows/build.yml)
![License](https://img.shields.io/badge/license-MIT-blue)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

# Alpaca Backtrader API

A Python library that provides seamless integration between Alpaca Markets and the Backtrader backtesting and live trading platform.

## Features

- **Unified Interface**: Same API for both backtesting and live trading
- **Historical Data**: Access to Alpaca's market data for backtesting
- **Live Trading**: Execute real trades through Alpaca's brokerage
- **Modern Backend**: Built on the official `alpaca-py` library (v0.16.0+)
- **Backward Compatible**: Drop-in replacement for existing code using v0.15.x
- **Python 3.10+**: Supports modern Python versions (3.10-3.13)

## Installation

### Using uv (Recommended)

```bash
# Add to existing project
uv add alpaca-backtrader-api

# Or install for development
git clone https://github.com/alpacahq/alpaca-backtrader-api.git
cd alpaca-backtrader-api
uv sync --dev
```

### Using pip

```bash
pip install alpaca-backtrader-api
```

## Quick Start

### 1. Basic Setup

```python
import backtrader as bt
from alpaca_backtrader_api import AlpacaBroker, AlpacaData, AlpacaStore

# Create Alpaca store
store = AlpacaStore(
    key_id='your_api_key',
    secret_key='your_secret_key',
    paper=True  # Set to False for live trading
)

# Create data feed
data = AlpacaData(
    dataname='AAPL',
    store=store,
    historical=True,  # For backtesting
    fromdate=datetime(2023, 1, 1),
    todate=datetime(2023, 12, 31)
)

# Create broker
broker = AlpacaBroker(store=store)

# Set up Backtrader cerebro
cerebro = bt.Cerebro()
cerebro.adddata(data)
cerebro.setbroker(broker)
```

### 2. Example Strategy

```python
class MyStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
    
    def next(self):
        if not self.position and self.data.close[0] > self.sma[0]:
            self.buy(size=100)
        elif self.position and self.data.close[0] < self.sma[0]:
            self.sell(size=100)

cerebro.addstrategy(MyStrategy)
cerebro.run()
```

### 3. Live Trading vs Backtesting

The only difference between backtesting and live trading is the `historical` parameter:

```python
# For backtesting
data = AlpacaData(dataname='AAPL', historical=True, fromdate=start, todate=end)

# For live trading  
data = AlpacaData(dataname='AAPL', historical=False)  # Uses real-time data
```

## Configuration

### Environment Variables

Set your Alpaca credentials as environment variables:

```bash
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"
export ALPACA_PAPER="true"  # or "false" for live trading
```

### Direct Configuration

```python
store = AlpacaStore(
    key_id="your_api_key",
    secret_key="your_secret_key", 
    paper=True,
    url="https://paper-api.alpaca.markets"  # Paper trading URL
)
```

## Development

### Setup Development Environment

```bash
git clone https://github.com/alpacahq/alpaca-backtrader-api.git
cd alpaca-backtrader-api
uv sync --dev
```

### Code Quality Tools

This project uses Ruff for linting, formatting, and import sorting:

```bash
# Lint code
uv run ruff check .

# Format code  
uv run ruff format .

# Fix linting issues automatically
uv run ruff check --fix .

# Type checking
uv run mypy alpaca_backtrader_api/
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=alpaca_backtrader_api --cov-report=html
```

## Migration from v0.15.x

**Good news!** Version 0.16.0 is fully backward compatible. No code changes are required.

The main improvements in v0.16.0:
- Migrated from deprecated `alpaca_trade_api` to official `alpaca-py` library
- Improved performance and reliability  
- Better error handling and debugging
- Modern development tooling with `uv` and `ruff`

For detailed migration information, see [MIGRATION.md](MIGRATION.md).

## Support

- **Documentation**: [GitHub README](https://github.com/alpacahq/alpaca-backtrader-api#readme)
- **Issues**: [GitHub Issues](https://github.com/alpacahq/alpaca-backtrader-api/issues)
- **Alpaca API**: [Alpaca Markets Documentation](https://alpaca.markets/docs/)
- **Backtrader**: [Backtrader Documentation](https://www.backtrader.com/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.
