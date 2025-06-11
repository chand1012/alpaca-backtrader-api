import collections
import threading
import time as _time
import traceback
from datetime import datetime, timedelta
from datetime import time as dtime
from enum import Enum

import backtrader as bt
import exchange_calendars
import pandas as pd
import pytz
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# Updated imports for alpaca-py
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
    TrailingStopOrderRequest,
)
from backtrader.metabase import MetaParams
from backtrader.utils.py3 import queue, with_metaclass
from dateutil.parser import parse as date_parse


NY = "America/New_York"


# Extend the exceptions to support extra cases
class AlpacaError(Exception):
    """Generic error class, catches Alpaca response errors"""

    def __init__(self, error_response):
        self.error_response = error_response
        msg = "Alpaca API returned error code {} ({}) ".format(
            error_response.get("code", "Unknown"),
            error_response.get("message", "Unknown error"),
        )

        super().__init__(msg)


class AlpacaRequestError(AlpacaError):
    def __init__(self):
        er = dict(code=599, message="Request Error", description="")
        super(self.__class__, self).__init__(er)


class AlpacaStreamError(AlpacaError):
    def __init__(self, content=""):
        er = dict(code=598, message="Failed Streaming", description=content)
        super(self.__class__, self).__init__(er)


class AlpacaTimeFrameError(AlpacaError):
    def __init__(self, content):
        er = dict(code=597, message="Not supported TimeFrame", description="")
        super(self.__class__, self).__init__(er)


class AlpacaNetworkError(AlpacaError):
    def __init__(self):
        er = dict(code=596, message="Network Error", description="")
        super(self.__class__, self).__init__(er)


class Granularity(Enum):
    Ticks = "ticks"
    Daily = "day"
    Minute = "minute"


class StreamingMethod(Enum):
    AccountUpdate = "account_update"
    Quote = "quote"
    MinuteAgg = "minute_agg"


class Streamer:
    conn = None

    def __init__(
        self,
        q,
        api_key="",
        api_secret="",
        instrument="",
        method: StreamingMethod = StreamingMethod.AccountUpdate,
        paper=False,
        data_feed="iex",
        *args,
        **kwargs,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.instrument = instrument
        self.method = method
        self.q = q
        self.data_feed = data_feed

    def run(self):
        # Create the streaming client
        self.conn = StockDataStream(
            api_key=self.api_key,
            secret_key=self.api_secret,
            paper=self.paper,
            feed=self.data_feed,
        )

        if self.method == StreamingMethod.AccountUpdate:
            # Trade updates are handled by trading client, not data stream
            pass
        elif self.method == StreamingMethod.MinuteAgg:
            self.conn.subscribe_bars(self.on_agg_min, self.instrument)
        elif self.method == StreamingMethod.Quote:
            self.conn.subscribe_quotes(self.on_quotes, self.instrument)

        # Run the stream
        self.conn.run()

    async def on_listen(self, conn, stream, msg):
        pass

    async def on_quotes(self, msg):
        # Convert to old format for compatibility
        data = {
            "time": msg.timestamp,
            "bid": msg.bid_price,
            "ask": msg.ask_price,
            "bid_size": msg.bid_size,
            "ask_size": msg.ask_size,
        }
        self.q.put(data)

    async def on_agg_min(self, msg):
        # Convert to old format for compatibility
        data = {
            "time": msg.timestamp,
            "open": msg.open,
            "high": msg.high,
            "low": msg.low,
            "close": msg.close,
            "volume": msg.volume,
        }
        self.q.put(data)

    async def on_account(self, msg):
        self.q.put(msg)

    async def on_trade(self, msg):
        self.q.put(msg)


class MetaSingleton(MetaParams):
    """Metaclass to make a metaclassed class a singleton"""

    def __init__(cls, name, bases, dct):
        super(MetaSingleton, cls).__init__(name, bases, dct)
        cls._singleton = None

    def __call__(cls, *args, **kwargs):
        if cls._singleton is None:
            cls._singleton = super(MetaSingleton, cls).__call__(*args, **kwargs)

        return cls._singleton


class AlpacaStore(with_metaclass(MetaSingleton, object)):
    """Singleton class wrapping to control the connections to Alpaca.

    Params:

      - ``key_id`` (default:``None``): Alpaca API key id

      - ``secret_key`` (default: ``None``): Alpaca API secret key

      - ``paper`` (default: ``False``): use the paper trading environment

      - ``account_tmout`` (default: ``10.0``): refresh period for account
        value/cash refresh
    """

    BrokerCls = None  # broker class will autoregister
    DataCls = None  # data class will auto register

    params = (
        ("key_id", ""),
        ("secret_key", ""),
        ("paper", False),
        ("account_tmout", 10.0),  # account balance refresh timeout
        ("data_feed", "iex"),  # data feed: iex or sip
    )

    _DTEPOCH = datetime(1970, 1, 1)

    @classmethod
    def getdata(cls, *args, **kwargs):
        """Returns ``DataCls`` with args, kwargs"""
        return cls.DataCls(*args, **kwargs)

    @classmethod
    def getbroker(cls, *args, **kwargs):
        """Returns broker with *args, **kwargs from registered ``BrokerCls``"""
        return cls.BrokerCls(*args, **kwargs)

    def __init__(self):
        super(AlpacaStore, self).__init__()

        self.notifs = collections.deque()  # store notifications for cerebro

        self._env = None  # reference to cerebro for general notifications
        self.broker = None  # broker instance
        self.datas = list()  # datas that have registered over start

        self._orders = collections.OrderedDict()  # map order.ref to oid
        self._ordersrev = collections.OrderedDict()  # map oid to order.ref
        self._transpend = collections.defaultdict(collections.deque)

        # Initialize the new alpaca-py clients
        self.trading_client = TradingClient(
            api_key=self.p.key_id, secret_key=self.p.secret_key, paper=self.p.paper
        )

        self.data_client = StockHistoricalDataClient(
            api_key=self.p.key_id, secret_key=self.p.secret_key
        )

        self._cash = 0.0
        self._value = 0.0
        self._evt_acct = threading.Event()

    def start(self, data=None, broker=None):
        # Datas require some processing to kickstart data reception
        if data is None and broker is None:
            self.cash = None
            return

        if data is not None:
            self._env = data._env
            # For datas simulate a queue with None to kickstart co
            self.datas.append(data)

            if self.broker is not None:
                self.broker.data_started(data)

        elif broker is not None:
            self.broker = broker
            self.streaming_events()
            self.broker_threads()

    def stop(self):
        # signal end of thread
        if self.broker is not None:
            self.q_ordercreate.put(None)
            self.q_orderclose.put(None)
            self.q_account.put(None)

    def put_notification(self, msg, *args, **kwargs):
        self.notifs.append((msg, args, kwargs))

    def get_notifications(self):
        """Return the pending "store" notifications"""
        self.notifs.append(None)  # put a mark / threads could still append
        return [x for x in iter(self.notifs.popleft, None)]

    # Alpaca supported granularities (updated for new API)
    _GRANULARITIES = {
        (bt.TimeFrame.Minutes, 1): TimeFrame.Minute,
        (bt.TimeFrame.Minutes, 5): TimeFrame(5, TimeFrameUnit.Minute),
        (bt.TimeFrame.Minutes, 15): TimeFrame(15, TimeFrameUnit.Minute),
        (bt.TimeFrame.Minutes, 30): TimeFrame(30, TimeFrameUnit.Minute),
        (bt.TimeFrame.Minutes, 60): TimeFrame.Hour,
        (bt.TimeFrame.Days, 1): TimeFrame.Day,
        (bt.TimeFrame.Weeks, 1): TimeFrame.Week,
        (bt.TimeFrame.Months, 1): TimeFrame.Month,
    }

    def get_positions(self):
        try:
            positions = self.trading_client.get_all_positions()
        except (APIError, Exception):
            return []
        return positions

    def get_granularity(self, timeframe, compression) -> Granularity:
        if timeframe == bt.TimeFrame.Ticks:
            return Granularity.Ticks
        if timeframe == bt.TimeFrame.Minutes:
            return Granularity.Minute
        if timeframe == bt.TimeFrame.Days:
            return Granularity.Daily
        return None

    def get_instrument(self, dataname):
        try:
            asset = self.trading_client.get_asset(dataname)
        except (APIError, Exception):
            return None
        return asset

    def streaming_events(self, tmout=None):
        q = queue.Queue()
        kwargs = {"q": q, "tmout": tmout}

        t = threading.Thread(target=self._t_streaming_listener, kwargs=kwargs)
        t.daemon = True
        t.start()

        t = threading.Thread(target=self._t_streaming_events, kwargs=kwargs)
        t.daemon = True
        t.start()
        return q

    def _t_streaming_listener(self, q, tmout=None):
        while True:
            trans = q.get()
            if hasattr(trans, "order"):
                self._transaction(trans.order)

    def _t_streaming_events(self, q, tmout=None):
        if tmout is not None:
            _time.sleep(tmout)

        # For trade updates, we'll use the trading client's stream
        # This is a simplified implementation - full implementation would require
        # setting up proper streaming for trade updates

    def candles(
        self,
        dataname,
        dtbegin,
        dtend,
        timeframe,
        compression,
        candleFormat,
        includeFirst,
    ):
        """
        Get historical bar data using the new alpaca-py API

        :param dataname: symbol name. e.g AAPL
        :param dtbegin: datetime start
        :param dtend: datetime end
        :param timeframe: bt.TimeFrame
        :param compression: distance between samples. e.g if 1 =>
                 get sample every day. if 3 => get sample every 3 days
        :param candleFormat: (bidask, midpoint, trades) - not used we get bars
        :param includeFirst:
        :return:
        """

        kwargs = locals().copy()
        kwargs.pop("self")
        kwargs["q"] = q = queue.Queue()
        t = threading.Thread(target=self._t_candles, kwargs=kwargs)
        t.daemon = True
        t.start()
        return q

    @staticmethod
    def iso_date(date_str):
        """
        this method will make sure that dates are formatted properly
        as with isoformat
        :param date_str:
        :return: YYYY-MM-DD date formatted
        """
        return date_parse(date_str).date().isoformat()

    def _t_candles(
        self,
        dataname,
        dtbegin,
        dtend,
        timeframe,
        compression,
        candleFormat,
        includeFirst,
        q,
    ):
        granularity: Granularity = self.get_granularity(timeframe, compression)
        dtbegin, dtend = self._make_sure_dates_are_initialized_properly(
            dtbegin, dtend, granularity
        )

        if granularity is None:
            e = AlpacaTimeFrameError("granularity is missing")
            q.put(e.error_response)
            return
        try:
            cdl = self.get_aggs_from_alpaca(
                dataname, dtbegin, dtend, granularity, compression
            )
        except AlpacaError as e:
            print(str(e))
            q.put(e.error_response)
            q.put(None)
            return
        except Exception:
            traceback.print_exc()
            q.put({"code": "error"})
            q.put(None)
            return

        # don't use dt.replace. use localize
        # (https://stackoverflow.com/a/1592837/2739124)
        cdl = cdl.loc[
            pytz.timezone(NY).localize(dtbegin)
            if not dtbegin.tzname()
            else dtbegin : pytz.timezone(NY).localize(dtend)
            if not dtend.tzname()
            else dtend
        ].dropna(subset=["high"])
        records = cdl.reset_index().to_dict("records")
        for r in records:
            r["time"] = r["timestamp"]
            q.put(r)
        q.put({})  # end of transmission

    def _make_sure_dates_are_initialized_properly(self, dtbegin, dtend, granularity):
        """
        dates may or may not be specified by the user.
        when they do, they are probably don't include NY timezome data
        also, when granularity is minute, we want to make sure we get data when
        market is opened. so if it doesn't - let's set end date to be last
        known minute with opened market.
        this method takes care of all these issues.
        :param dtbegin:
        :param dtend:
        :param granularity:
        :return:
        """
        if not dtend:
            dtend = pd.Timestamp("now", tz=NY)
        else:
            dtend = (
                pd.Timestamp(pytz.timezone("UTC").localize(dtend))
                if not dtend.tzname()
                else dtend
            )
        if granularity == Granularity.Minute:
            calendar = exchange_calendars.get_calendar(name="NYSE")
            while not calendar.is_open_on_minute(dtend.ceil(freq="T")):
                dtend = dtend.replace(hour=15, minute=59, second=0, microsecond=0)
                dtend -= timedelta(days=1)
        if not dtbegin:
            days = 30 if granularity == Granularity.Daily else 3
            delta = timedelta(days=days)
            dtbegin = dtend - delta
        else:
            dtbegin = (
                pd.Timestamp(pytz.timezone("UTC").localize(dtbegin))
                if not dtbegin.tzname()
                else dtbegin
            )
        while dtbegin > dtend:
            # if we start the script during market hours we could get this
            # situation. this resolves that.
            dtbegin -= timedelta(days=1)
        return dtbegin.astimezone(NY), dtend.astimezone(NY)

    def get_aggs_from_alpaca(
        self, dataname, start, end, granularity: Granularity, compression
    ):
        """
        Get aggregated bar data using the new alpaca-py API
        """

        def _granularity_to_timeframe(granularity, compression):
            if granularity in [Granularity.Minute, Granularity.Ticks]:
                if compression == 1:
                    return TimeFrame.Minute
                return TimeFrame(compression, TimeFrameUnit.Minute)
            if granularity == Granularity.Daily:
                if compression == 1:
                    return TimeFrame.Day
                return TimeFrame(compression, TimeFrameUnit.Day)
            # default to day if not configured properly
            return TimeFrame.Day

        def _get_bars_request(symbol, timeframe, start_time, end_time):
            """Create bars request for the new API"""
            return StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_time,
                end=end_time,
            )

        def _iterate_api_calls():
            """
            Get historical bars with pagination support
            """
            timeframe = _granularity_to_timeframe(granularity, compression)

            # Create the request
            request = _get_bars_request(dataname, timeframe, start, end)

            # Get bars from the data client
            bars_response = self.data_client.get_stock_bars(request)

            if dataname in bars_response:
                bars_df = bars_response[dataname].df
                return bars_df
            return pd.DataFrame()

        def _clear_out_of_market_hours(df):
            """
            only interested in samples between 9:30, 16:00 NY time
            """
            return df.between_time("09:30", "16:00")

        def _drop_early_samples(df):
            """
            samples from server don't start at 9:30 NY time
            let's drop earliest samples
            """
            for i, b in df.iterrows():
                if i.time() >= dtime(9, 30):
                    return df[i:]

        def _resample(df):
            """
            samples returned with certain window size (1 day, 1 minute) user
            may want to work with different window size (5min)
            """

            if granularity == Granularity.Minute:
                sample_size = f"{compression}Min"
            else:
                sample_size = f"{compression}D"
            df = df.resample(sample_size).agg(
                collections.OrderedDict(
                    [
                        ("open", "first"),
                        ("high", "max"),
                        ("low", "min"),
                        ("close", "last"),
                        ("volume", "sum"),
                    ]
                )
            )
            if granularity == Granularity.Minute:
                return df.between_time("09:30", "16:00")
            return df

        response = _iterate_api_calls()
        cdl = response

        if granularity == Granularity.Minute:
            cdl = _clear_out_of_market_hours(cdl)
            cdl = _drop_early_samples(cdl)
        if compression != 1:
            response = _resample(cdl)
        else:
            response = cdl
        response = response.dropna()
        response = response[~response.index.duplicated()]
        return response

    def streaming_prices(self, dataname, timeframe, tmout=None, data_feed="iex"):
        q = queue.Queue()
        kwargs = {
            "q": q,
            "dataname": dataname,
            "timeframe": timeframe,
            "data_feed": data_feed,
            "tmout": tmout,
        }
        t = threading.Thread(target=self._t_streaming_prices, kwargs=kwargs)
        t.daemon = True
        t.start()
        return q

    def _t_streaming_prices(self, dataname, timeframe, q, tmout, data_feed):
        if tmout is not None:
            _time.sleep(tmout)

        if timeframe == bt.TimeFrame.Ticks:
            method = StreamingMethod.Quote
        elif timeframe == bt.TimeFrame.Minutes:
            method = StreamingMethod.MinuteAgg
        else:
            method = StreamingMethod.MinuteAgg

        streamer = Streamer(
            q,
            api_key=self.p.key_id,
            api_secret=self.p.secret_key,
            instrument=dataname,
            method=method,
            paper=self.p.paper,
            data_feed=data_feed,
        )

        streamer.run()

    def get_cash(self):
        return self._cash

    def get_value(self):
        return self._value

    _ORDEREXECS = {
        bt.Order.Market: OrderType.MARKET,
        bt.Order.Limit: OrderType.LIMIT,
        bt.Order.Stop: OrderType.STOP,
        bt.Order.StopLimit: OrderType.STOP_LIMIT,
        bt.Order.StopTrail: OrderType.TRAILING_STOP,
    }

    def broker_threads(self):
        self.q_account = queue.Queue()
        self.q_account.put(True)  # force an immediate update
        t = threading.Thread(target=self._t_account)
        t.daemon = True
        t.start()

        self.q_ordercreate = queue.Queue()
        t = threading.Thread(target=self._t_order_create)
        t.daemon = True
        t.start()

        self.q_orderclose = queue.Queue()
        t = threading.Thread(target=self._t_order_cancel)
        t.daemon = True
        t.start()

        # Wait once for the values to be set
        self._evt_acct.wait(self.p.account_tmout)

    def _t_account(self):
        while True:
            try:
                msg = self.q_account.get(timeout=self.p.account_tmout)
                if msg is None:
                    break  # end of thread
            except queue.Empty:  # tmout -> time to refresh
                pass

            try:
                account = self.trading_client.get_account()
            except Exception as e:
                self.put_notification(e)
                continue

            try:
                self._cash = float(account.cash)
                self._value = float(account.portfolio_value)
            except (KeyError, AttributeError):
                pass

            self._evt_acct.set()

    def order_create(self, order, stopside=None, takeside=None, **kwargs):
        # Determine order side
        side = OrderSide.BUY if order.isbuy() else OrderSide.SELL

        # Get symbol name
        symbol = order.data._name if order.data._name else order.data._dataname

        # Get quantity
        qty = abs(int(order.created.size))

        # Create the appropriate order request based on order type
        if order.exectype == bt.Order.Market:
            order_request = MarketOrderRequest(
                symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.GTC
            )
        elif order.exectype == bt.Order.Limit:
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                limit_price=order.created.price,
            )
        elif order.exectype == bt.Order.Stop:
            order_request = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                stop_price=order.created.pricelimit,
            )
        elif order.exectype == bt.Order.StopLimit:
            order_request = StopLimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                limit_price=order.created.price,
                stop_price=order.created.pricelimit,
            )
        elif order.exectype == bt.Order.StopTrail:
            if order.trailpercent and order.trailamount:
                raise Exception(
                    "You can't create trailing stop order with "
                    "both TrailPrice and TrailPercent. choose one"
                )

            trail_params = {}
            if order.trailpercent:
                trail_params["trail_percent"] = order.trailpercent
            elif order.trailamount:
                trail_params["trail_price"] = order.trailamount
            else:
                raise Exception(
                    "You must provide either trailpercent or "
                    "trailamount when creating StopTrail order"
                )

            order_request = TrailingStopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                **trail_params,
            )
        else:
            raise ValueError(f"Unsupported order type: {order.exectype}")

        # Handle bracket orders (simplified implementation)
        # Note: Bracket orders would need more complex handling with the new API
        # This is a placeholder for future implementation

        self.q_ordercreate.put((order.ref, order_request))
        return order

    def _t_order_create(self):
        def _check_if_transaction_occurred(order_id):
            # a transaction may have happened and was stored. if so let's
            # process it
            tpending = self._transpend[order_id]
            tpending.append(None)  # eom marker
            while True:
                trans = tpending.popleft()
                if trans is None:
                    break
                self._process_transaction(order_id, trans)

        while True:
            try:
                if self.q_ordercreate.empty():
                    continue
                msg = self.q_ordercreate.get()
                if msg is None:
                    continue
                oref, order_request = msg
                try:
                    order_response = self.trading_client.submit_order(order_request)
                except Exception as e:
                    self.put_notification(e)
                    self.broker._reject(oref)
                    continue

                try:
                    oid = order_response.id
                except Exception:
                    self.put_notification("Error getting order ID from response")
                    self.broker._reject(oref)
                    continue

                if isinstance(order_request, MarketOrderRequest):
                    self.broker._accept(oref)  # taken immediately

                self._orders[oref] = oid
                self._ordersrev[oid] = oref  # maps ids to backtrader order
                _check_if_transaction_occurred(oid)

                # Handle bracket order legs
                if hasattr(order_response, "legs") and order_response.legs:
                    index = 1
                    for leg in order_response.legs:
                        self._orders[oref + index] = leg.id
                        self._ordersrev[leg.id] = oref + index
                        _check_if_transaction_occurred(leg.id)
                        index += 1

                self.broker._submit(oref)  # inside it submits the legs too
                if isinstance(order_request, MarketOrderRequest):
                    self.broker._accept(oref)  # taken immediately

            except Exception as e:
                print(str(e))

    def order_cancel(self, order):
        self.q_orderclose.put(order.ref)
        return order

    def _t_order_cancel(self):
        while True:
            oref = self.q_orderclose.get()
            if oref is None:
                break

            oid = self._orders.get(oref, None)
            if oid is None:
                continue  # the order is no longer there
            try:
                self.trading_client.cancel_order_by_id(oid)
            except Exception as e:
                self.put_notification(f"Order not cancelled: {oid}, {e}")
                continue

            self.broker._cancel(oref)

    _X_ORDER_CREATE = (
        "new",
        "accepted",
        "pending_new",
        "accepted_for_bidding",
    )

    def _transaction(self, trans):
        # Invoked from Streaming Events. May actually receive an event for an
        # oid which has not yet been returned after creating an order. Hence
        # store if not yet seen, else forward to processer

        oid = trans.get("id") if isinstance(trans, dict) else getattr(trans, "id", None)

        if not self._ordersrev.get(oid, False):
            self._transpend[oid].append(trans)
        self._process_transaction(oid, trans)

    _X_ORDER_FILLED = (
        "partially_filled",
        "filled",
    )

    def _process_transaction(self, oid, trans):
        try:
            oref = self._ordersrev[oid]
        except KeyError:
            return

        # Get transaction status
        status = (
            trans.get("status")
            if isinstance(trans, dict)
            else getattr(trans, "status", None)
        )

        if status in self._X_ORDER_FILLED:
            size = (
                trans.get("filled_qty", 0)
                if isinstance(trans, dict)
                else getattr(trans, "filled_qty", 0)
            )
            price = (
                trans.get("filled_avg_price", 0)
                if isinstance(trans, dict)
                else getattr(trans, "filled_avg_price", 0)
            )
            self.broker._fill(oref, size, price, 0)

        elif status == "cancelled":
            self.broker._cancel(oref)

        elif status == "rejected":
            self.broker._reject(oref)
