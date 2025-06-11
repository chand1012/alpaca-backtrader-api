#!/usr/bin/env python3
"""
Example usage of the Alpaca Backtrader API (version 0.16.0)

This example demonstrates the updated alpaca-backtrader-api which now uses
the modern alpaca-py library while maintaining full backward compatibility.
Your existing code continues to work without any changes.
"""

import os
from datetime import datetime, timedelta

import backtrader as bt

from alpaca_backtrader_api import AlpacaStore


# Get API keys from environment variables for security
API_KEY = os.getenv("ALPACA_API_KEY", "your_api_key_here")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "your_secret_key_here")
PAPER = True  # Set to False for live trading


class SimpleMovingAverageStrategy(bt.Strategy):
    """
    Simple moving average crossover strategy
    """

    params = (
        ("fast_period", 10),
        ("slow_period", 30),
        ("size", 100),
    )

    def __init__(self):
        # Calculate moving averages
        self.fast_sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.fast_period
        )
        self.slow_sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.slow_period
        )

        # Create crossover signal
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

        # Track order status
        self.order = None

    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            return  # Order is still pending

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}")
            else:
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        self.order = None

    def notify_trade(self, trade):
        """Handle trade notifications"""
        if trade.isclosed:
            self.log(f"TRADE PROFIT: {trade.pnl:.2f}")

    def log(self, text, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}: {text}")

    def next(self):
        """Strategy logic"""
        # Don't trade if we have a pending order
        if self.order:
            return

        # Buy signal: fast MA crosses above slow MA
        if self.crossover[0] > 0 and not self.position:
            self.log(f"BUY CREATE, Price: {self.data.close[0]:.2f}")
            self.order = self.buy(size=self.params.size)

        # Sell signal: fast MA crosses below slow MA
        elif self.crossover[0] < 0 and self.position:
            self.log(f"SELL CREATE, Price: {self.data.close[0]:.2f}")
            self.order = self.sell(size=self.params.size)


def run_backtest():
    """Run a historical backtest"""
    print("Running Historical Backtest...")

    cerebro = bt.Cerebro()

    # Create the Alpaca store
    store = AlpacaStore(key_id=API_KEY, secret_key=SECRET_KEY, paper=PAPER)

    # Create historical data feed
    data = store.getdata(
        dataname="AAPL",
        historical=True,
        fromdate=datetime.now() - timedelta(days=30),
        todate=datetime.now(),
        timeframe=bt.TimeFrame.Minutes,
        compression=5,
    )

    cerebro.adddata(data)
    cerebro.addstrategy(SimpleMovingAverageStrategy)

    # Set initial cash
    cerebro.broker.setcash(100000.0)

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")


def run_live_trading():
    """Run live trading (paper trading)"""
    print("Running Live Paper Trading...")

    cerebro = bt.Cerebro()

    # Create the Alpaca store
    store = AlpacaStore(key_id=API_KEY, secret_key=SECRET_KEY, paper=PAPER)

    # Create live data feed
    data = store.getdata(
        dataname="AAPL", historical=False, timeframe=bt.TimeFrame.Minutes, compression=1
    )

    # Use the Alpaca broker for live trading
    broker = store.getbroker()
    cerebro.setbroker(broker)

    cerebro.adddata(data)
    cerebro.addstrategy(SimpleMovingAverageStrategy)

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    print("Starting live trading... Press Ctrl+C to stop")

    try:
        cerebro.run()
    except KeyboardInterrupt:
        print("\nStopping live trading...")

    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alpaca Backtrader Example")
    parser.add_argument(
        "--live", action="store_true", help="Run live trading instead of backtest"
    )

    args = parser.parse_args()

    if API_KEY == "your_api_key_here" or SECRET_KEY == "your_secret_key_here":
        print("Please set your Alpaca API credentials:")
        print("export ALPACA_API_KEY='your_actual_api_key'")
        print("export ALPACA_SECRET_KEY='your_actual_secret_key'")
        exit(1)

    if args.live:
        run_live_trading()
    else:
        run_backtest()
