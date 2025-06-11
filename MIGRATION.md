# Migration Guide: Version 0.16.0 Update

This guide helps you understand the changes in version 0.16.0 of `alpaca-backtrader-api` and how to take advantage of the new development tools.

## Overview

Version 0.16.0 represents a **non-breaking update** that migrates from the deprecated `alpaca_trade_api` library to the modern `alpaca-py` library. **Your existing code will continue to work without any changes.**

### What's New in 0.16.0

- **Backend Migration**: Uses the officially supported `alpaca-py` library
- **Modern Package Management**: Built with `pyproject.toml` and `uv`
- **Improved Performance**: Better stability and faster data retrieval
- **Enhanced Developer Experience**: Faster dependency management and builds
- **Full Backward Compatibility**: All existing code continues to work

## For End Users

### No Action Required

If you're using `alpaca-backtrader-api` in your trading strategies, **no changes are needed**. Simply update to the latest version:

```bash
pip install --upgrade alpaca-backtrader-api
```

### Your Code Remains the Same

All existing code continues to work exactly as before:

```python
# This code works identically in 0.16.0
import backtrader as bt
from alpaca_backtrader_api import AlpacaStore

store = AlpacaStore(
    key_id='your_api_key',
    secret_key='your_secret_key',
    paper=True
)

data = store.getdata(dataname='AAPL', historical=True)
broker = store.getbroker()
# ... rest of your strategy code unchanged
```

## For Developers and Contributors

### Package Management Migration

The project now uses modern Python packaging standards:

**Old Approach (≤ 0.15.0):**
- `setup.py` and `setup.cfg` for package configuration
- `requirements.txt` for dependency management
- Manual virtual environment setup

**New Approach (≥ 0.16.0):**
- `pyproject.toml` for all package configuration
- `uv` for fast dependency management
- Integrated development tools

### Setting Up Development Environment

#### Option 1: Using uv (Recommended)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup the project
git clone https://github.com/alpacahq/alpaca-backtrader-api.git
cd alpaca-backtrader-api

# Install all dependencies (including dev dependencies)
uv sync --dev

# Run tests
uv run pytest

# Run example
uv run python example.py
```

#### Option 2: Using Traditional pip

```bash
# Clone the repository
git clone https://github.com/alpacahq/alpaca-backtrader-api.git
cd alpaca-backtrader-api

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Run tests
pytest
```

### Development Workflow

#### Code Formatting and Quality

```bash
# Format code with Ruff
uv run ruff format .

# Type checking with mypy
uv run mypy alpaca_backtrader_api/

# Run all quality checks
uv run ruff format . && uv run mypy alpaca_backtrader_api/
```

#### Testing

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=alpaca_backtrader_api

# Run tests with coverage report
uv run pytest --cov=alpaca_backtrader_api --cov-report=html
```

#### Building the Package

```bash
# Build wheel and source distribution
uv build

# Build specific format
uv build --wheel
uv build --sdist
```

### Package Configuration

The `pyproject.toml` file now contains all package metadata:

```toml
[project]
name = "alpaca-backtrader-api"
version = "0.16.0"
description = "Alpaca trading API integration for Backtrader"
dependencies = [
    "alpaca-py>=0.13.0",
    "backtrader>=1.9.74.123",
    # ... other dependencies
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "ruff>=0.0.255",
    "mypy>=0.991",
    # ... dev dependencies
]
```

## API Library Migration Details

### What Changed Under the Hood

The migration from `alpaca_trade_api` to `alpaca-py` includes:

1. **Client Structure**: 
   - Old: `alpaca_trade_api.REST`
   - New: `alpaca.trading.client.TradingClient` + `alpaca.data.historical.StockHistoricalDataClient`

2. **Request Objects**:
   - Old: Dictionary-based parameters
   - New: Structured request objects (`MarketOrderRequest`, `LimitOrderRequest`, etc.)

3. **Error Handling**:
   - Old: `alpaca_trade_api.rest.APIError`
   - New: `alpaca.common.exceptions.APIError`

4. **Streaming**:
   - Old: `alpaca_trade_api.stream.Stream`
   - New: `alpaca.data.live.StockDataStream`

### Maintained Compatibility

Despite these internal changes, the public API remains identical:

- All `AlpacaStore` parameters work the same
- All `AlpacaData` parameters work the same
- All `AlpacaBroker` functionality works the same
- All order types are supported
- All timeframes and data feeds work the same

## Performance Improvements

### Speed Improvements

- **Dependency Installation**: ~10x faster with `uv`
- **Package Building**: ~5x faster with modern build backend
- **API Calls**: Improved reliability with `alpaca-py`
- **Development Setup**: Faster environment creation and management

### Memory and Stability

- **Better Error Handling**: More robust connection management
- **Memory Usage**: Optimized data structures in `alpaca-py`
- **Connection Pooling**: Improved network request handling

## Troubleshooting

### Common Issues

1. **Import Errors After Update**
   ```bash
   # Clear pip cache and reinstall
   pip cache purge
   pip uninstall alpaca-backtrader-api
   pip install alpaca-backtrader-api
   ```

2. **Development Setup Issues**
   ```bash
   # Ensure uv is properly installed
   uv --version
   
   # If using pip, ensure you have the latest version
   pip install --upgrade pip setuptools wheel
   ```

3. **Type Checking Issues**
   ```bash
   # Install type stubs if needed
   uv add --dev types-requests types-pytz
   ```

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/alpacahq/alpaca-backtrader-api/issues)
- **Discussions**: [GitHub Discussions](https://github.com/alpacahq/alpaca-backtrader-api/discussions)
- **Alpaca Support**: [Alpaca Documentation](https://alpaca.markets/docs/)

## Benefits of the Update

### For Users
- **Reliability**: Better connection handling and error recovery
- **Performance**: Faster data retrieval and processing
- **Future-Proof**: Built on actively maintained libraries
- **No Changes Required**: Seamless update experience

### For Developers
- **Faster Development**: Quick setup with `uv`
- **Modern Tooling**: Integrated code formatting, testing, and building
- **Better CI/CD**: Faster builds and more reliable testing
- **Enhanced Debugging**: Better error messages and logging

## Migration Checklist

### For Users
- [ ] Update package: `pip install --upgrade alpaca-backtrader-api`
- [ ] Test existing strategies (should work without changes)
- [ ] Verify API credentials and permissions still work

### For Contributors/Developers
- [ ] Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ ] Clone updated repository
- [ ] Run `uv sync --dev` to setup development environment
- [ ] Verify tests pass: `uv run pytest`
- [ ] Update any CI/CD scripts to use `uv` (optional but recommended)

## Future Roadmap

With the foundation now modernized, upcoming features may include:

- Enhanced streaming data capabilities
- Additional order types supported by `alpaca-py`
- Improved error handling and retry logic
- Better integration with Alpaca's latest API features
- Performance optimizations specific to the new backend

## Conclusion

Version 0.16.0 provides a solid foundation for future development while maintaining complete backward compatibility. Users can update with confidence, knowing their existing code will continue to work, while developers benefit from modern tooling and improved performance.

The migration to `alpaca-py` and modern packaging standards ensures the project remains maintainable and aligned with Python ecosystem best practices. 
