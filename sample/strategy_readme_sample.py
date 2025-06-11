# This is the example code from the repo's README
from datetime import datetime

import backtrader as bt

import alpaca_backtrader_api


# Your credentials here
ALPACA_API_KEY = "<key_id>"
ALPACA_SECRET_KEY = "<secret_key>"


"""
You have 3 options:
 - backtest (IS_BACKTEST=True, IS_LIVE=False)
 - paper trade (IS_BACKTEST=False, IS_LIVE=False)
 - live trade (IS_BACKTEST=False, IS_LIVE=True)
"""
IS_BACKTEST = False
IS_LIVE = False
symbol = "AAPL"


class SmaCross(bt.SignalStrategy):
    def __init__(self):
        sma1, sma2 = bt.ind.SMA(period=10), bt.ind.SMA(period=30)
        crossover = bt.ind.CrossOver(sma1, sma2)
        self.signal_add(bt.SIGNAL_LONG, crossover)


if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross)

    store = alpaca_backtrader_api.AlpacaStore(
        key_id=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        paper=not IS_LIVE,
    )

    DataFactory = store.getdata  # or use alpaca_backtrader_api.AlpacaData
    if IS_BACKTEST:
        data0 = DataFactory(
            dataname=symbol,
            historical=True,
            fromdate=datetime(2015, 1, 1),
            timeframe=bt.TimeFrame.Days,
            data_feed="iex",
        )
    else:
        data0 = DataFactory(
            dataname=symbol,
            historical=False,
            timeframe=bt.TimeFrame.Days,
            data_feed="iex",
        )
        # or just alpaca_backtrader_api.AlpacaBroker()
        broker = store.getbroker()
        cerebro.setbroker(broker)
    cerebro.adddata(data0)

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue()}")
    cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue()}")
    cerebro.plot()
