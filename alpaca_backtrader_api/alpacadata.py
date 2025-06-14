
from datetime import timedelta

import backtrader as bt
import pandas as pd
from backtrader import date2num, num2date
from backtrader.feed import DataBase
from backtrader.utils.py3 import queue, with_metaclass

from alpaca_backtrader_api import alpacastore


class MetaAlpacaData(DataBase.__class__):
    def __init__(cls, name, bases, dct):
        """
        Class has already been created ... register
        """
        # Initialize the class
        super(MetaAlpacaData, cls).__init__(name, bases, dct)

        # Register with the store
        alpacastore.AlpacaStore.DataCls = cls


class AlpacaData(with_metaclass(MetaAlpacaData, DataBase)):
    """
    Alpaca Data Feed.

    Params:

      - ``qcheck`` (default: ``0.5``)

        Time in seconds to wake up if no data is received to give a chance to
        resample/replay packets properly and pass notifications up the chain

      - ``historical`` (default: ``False``)

        If set to ``True`` the data feed will stop after doing the first
        download of data.

        The standard data feed parameters ``fromdate`` and ``todate`` will be
        used as reference.

        The data feed will make multiple requests if the requested duration is
        larger than the one allowed by Alpaca given the timeframe/compression
        chosen for the data.

      - ``backfill_start`` (default: ``True``)

        Perform backfilling at the start. The maximum possible historical data
        will be fetched in a single request.

      - ``backfill`` (default: ``True``)

        Perform backfilling after a disconnection/reconnection cycle. The gap
        duration will be used to download the smallest possible amount of data

      - ``backfill_from`` (default: ``None``)

        An additional data source can be passed to do an initial layer of
        backfilling. Once the data source is depleted and if requested,
        backfilling from Alpaca will take place. This is ideally meant to backfill
        from already stored sources like a file on disk, but not limited to.

      - ``bidask`` (default: ``True``) - THIS IS NOT USED. WE GET BARS NOT
                                         QUOTES/TICKS FOR HISTORIC PRICES

        If ``True``, then the historical/backfilling requests will request
        bid/ask prices from the server

        If ``False``, then *midpoint* will be requested

      - ``useask`` (default: ``False``)

        If ``True`` the *ask* part of the *bidask* prices will be used instead
        of the default use of *bid*

      - ``includeFirst`` (default: ``True``)

        Influence the delivery of the 1st bar of a historical/backfilling
        request by setting the parameter directly to the Alpaca API calls

      - ``reconnect`` (default: ``True``)

        Reconnect when network connection is down

      - ``reconnections`` (default: ``-1``)

        Number of times to attempt reconnections: ``-1`` means forever

      - ``reconntimeout`` (default: ``5.0``)

        Time in seconds to wait in between reconnection attemps

    This data feed supports the following mapping of ``timeframe`` and
    ``compression``, which comply with the definitions in the Alpaca API
    Developer's Guide::

        (TimeFrame.Minutes, 1): '1Min',
        (TimeFrame.Minutes, 5): '5Min',
        (TimeFrame.Minutes, 15): '15Min',
        (TimeFrame.Minutes, 30): '30Min',
        (TimeFrame.Minutes, 60): '1Hour',
        (TimeFrame.Days, 1): '1Day',
        (TimeFrame.Weeks, 1): '1Week',
        (TimeFrame.Months, 1): '1Month',

    Any other combination will be rejected
    """

    params = (
        ("qcheck", 0.5),
        ("historical", False),  # do backfilling at the start
        ("backfill_start", True),  # do backfilling at the start
        ("backfill", True),  # do backfilling when reconnecting
        ("backfill_from", None),  # additional data source to do backfill from
        ("bidask", True),
        ("useask", False),
        ("includeFirst", True),
        ("reconnect", True),
        ("reconnections", -1),  # forever
        ("reconntimeout", 5.0),
        ("data_feed", "iex"),  # options iex/sip for pro
    )

    _store = alpacastore.AlpacaStore

    # States for the Finite State Machine in _load
    _ST_FROM, _ST_START, _ST_LIVE, _ST_HISTORBACK, _ST_OVER = range(5)

    _TOFFSET = timedelta()

    def _timeoffset(self):
        # Effective way to overcome the non-notification?
        return self._TOFFSET

    def islive(self):
        """
        Returns ``True`` to notify ``Cerebro`` that preloading and runonce
        should be deactivated
        """
        return True

    def __init__(self, **kwargs):
        self.o = self._store(**kwargs)
        self._candleFormat = "bidask" if self.p.bidask else "midpoint"
        self._timeframe = self.p.timeframe
        self.do_qcheck(True, 0)
        if self._timeframe not in [
            bt.TimeFrame.Ticks,
            bt.TimeFrame.Minutes,
            bt.TimeFrame.Days,
        ]:
            raise Exception(
                f"Unsupported time frame: {bt.TimeFrame.TName(self._timeframe)}"
            )

    def setenvironment(self, env):
        """
        Receives an environment (cerebro) and passes it over to the store it
        belongs to
        """
        super(AlpacaData, self).setenvironment(env)
        env.addstore(self.o)

    def start(self):
        """
        Starts the Alpaca connection and gets the real contract and
        contractdetails if it exists
        """
        super(AlpacaData, self).start()

        # Create attributes as soon as possible
        self._statelivereconn = False  # if reconnecting in live state
        self._storedmsg = dict()  # keep pending live message (under None)
        self.qlive = queue.Queue()
        self._state = self._ST_OVER

        # Kickstart store and get queue to wait on
        self.o.start(data=self)

        # check if the granularity is supported
        otf = self.o.get_granularity(self._timeframe, self._compression)
        if otf is None:
            self.put_notification(self.NOTSUPPORTED_TF)
            self._state = self._ST_OVER
            return

        self.contractdetails = cd = self.o.get_instrument(self.p.dataname)
        if cd is None:
            self.put_notification(self.NOTSUBSCRIBED)
            self._state = self._ST_OVER
            return

        if self.p.backfill_from is not None:
            self._state = self._ST_FROM
            self.p.backfill_from.setenvironment(self._env)
            self.p.backfill_from._start()
        else:
            self._start_finish()
            self._state = self._ST_START  # initial state for _load
            self._st_start()

        self._reconns = 0

    def _st_start(self, instart=True, tmout=None):
        if self.p.historical:
            self.put_notification(self.DELAYED)
            dtend = None
            if self.todate < float("inf"):
                dtend = num2date(self.todate)

            dtbegin = None
            if self.fromdate > float("-inf"):
                dtbegin = num2date(self.fromdate)

            self.qhist = self.o.candles(
                self.p.dataname,
                dtbegin,
                dtend,
                self._timeframe,
                self._compression,
                candleFormat=self._candleFormat,
                includeFirst=self.p.includeFirst,
            )

            self._state = self._ST_HISTORBACK
            return True
        self.qlive = self.o.streaming_prices(
            self.p.dataname, self.p.timeframe, tmout=tmout, data_feed=self.p.data_feed
        )
        if instart:
            self._statelivereconn = self.p.backfill_start
        else:
            self._statelivereconn = self.p.backfill

        if self._statelivereconn:
            self.put_notification(self.DELAYED)

        self._state = self._ST_LIVE
        if instart:
            self._reconns = self.p.reconnections

        return True  # no return before - implicit continue

    def stop(self):
        """
        Stops and tells the store to stop
        """
        super(AlpacaData, self).stop()
        self.o.stop()

    def haslivedata(self):
        return bool(self._storedmsg or self.qlive)  # do not return the objs

    def _load(self):
        if self._state == self._ST_OVER:
            return False

        while True:
            if self._state == self._ST_LIVE:
                try:
                    msg = self._storedmsg.pop(None, None) or self.qlive.get(
                        timeout=self._qcheck
                    )
                except queue.Empty:
                    return None  # indicate timeout situation
                if msg is None:  # Conn broken during historical/backfilling
                    self.put_notification(self.CONNBROKEN)
                    # Try to reconnect
                    if not self.p.reconnect or self._reconns == 0:
                        # Can no longer reconnect
                        self.put_notification(self.DISCONNECTED)
                        self._state = self._ST_OVER
                        return False  # failed

                    self._reconns -= 1
                    self._st_start(instart=False, tmout=self.p.reconntimeout)
                    continue

                if "code" in msg:
                    self.put_notification(self.CONNBROKEN)
                    code = msg["code"]
                    if code not in [599, 598, 596]:
                        self.put_notification(self.DISCONNECTED)
                        self._state = self._ST_OVER
                        return False  # failed

                    if not self.p.reconnect or self._reconns == 0:
                        # Can no longer reconnect
                        self.put_notification(self.DISCONNECTED)
                        self._state = self._ST_OVER
                        return False  # failed

                    # Can reconnect
                    self._reconns -= 1
                    self._st_start(instart=False, tmout=self.p.reconntimeout)
                    continue

                self._reconns = self.p.reconnections

                # Process the message according to expected return type
                if not self._statelivereconn:
                    if self._laststatus != self.LIVE:
                        if self.qlive.qsize() <= 1:  # very short live queue
                            self.put_notification(self.LIVE)
                    if self.p.timeframe == bt.TimeFrame.Ticks:
                        ret = self._load_tick(msg)
                    elif self.p.timeframe == bt.TimeFrame.Minutes:
                        ret = self._load_agg(msg)
                    else:
                        # might want to act differently in the future
                        ret = self._load_agg(msg)
                    if ret:
                        return True

                    # could not load bar ... go and get new one
                    continue

                # Fall through to processing reconnect - try to backfill
                self._storedmsg[None] = msg  # keep the msg

                # else do a backfill
                if self._laststatus != self.DELAYED:
                    self.put_notification(self.DELAYED)

                dtend = None
                if len(self) > 1:
                    # len == 1 ... forwarded for the 1st time
                    dtbegin = self.datetime.datetime(-1)
                elif self.fromdate > float("-inf"):
                    dtbegin = num2date(self.fromdate)
                else:  # 1st bar and no begin set
                    # passing None to fetch max possible in 1 request
                    dtbegin = None

                dtend = pd.Timestamp(msg["time"], unit="ns")

                self.qhist = self.o.candles(
                    self.p.dataname,
                    dtbegin,
                    dtend,
                    self._timeframe,
                    self._compression,
                    candleFormat=self._candleFormat,
                    includeFirst=self.p.includeFirst,
                )

                self._state = self._ST_HISTORBACK
                self._statelivereconn = False  # no longer in live
                continue

            if self._state == self._ST_HISTORBACK:
                msg = self.qhist.get()
                if msg is None:  # Conn broken during historical/backfilling
                    # Situation not managed. Simply bail out
                    self.put_notification(self.DISCONNECTED)
                    self._state = self._ST_OVER
                    return False  # error management cancelled the queue

                if "code" in msg:  # Error
                    self.put_notification(self.NOTSUBSCRIBED)
                    self.put_notification(self.DISCONNECTED)
                    self._state = self._ST_OVER
                    return False

                if msg:
                    if self._load_history(msg):
                        return True  # loading worked

                    continue  # not loaded ... date may have been seen
                # End of histdata
                if self.p.historical:  # only historical
                    self.put_notification(self.DISCONNECTED)
                    self._state = self._ST_OVER
                    return False  # end of historical

                # Live is also wished - go for it
                self._state = self._ST_LIVE
                continue

            if self._state == self._ST_FROM:
                if not self.p.backfill_from.next():
                    # additional data source is consumed
                    self._state = self._ST_START
                    continue

                # copy lines of the same name
                for alias in self.lines.getlinealiases():
                    lsrc = getattr(self.p.backfill_from.lines, alias)
                    ldst = getattr(self.lines, alias)

                    ldst[0] = lsrc[0]

                return True

            if self._state == self._ST_START:
                if not self._st_start(instart=False):
                    self._state = self._ST_OVER
                    return False

    def _load_tick(self, msg):
        dtobj = pd.Timestamp(msg["time"], unit="ns")
        dt = date2num(dtobj)
        if dt <= self.lines.datetime[-1]:
            return False  # time already seen

        # Common fields
        self.lines.datetime[0] = dt
        self.lines.volume[0] = 0.0
        self.lines.openinterest[0] = 0.0

        # Put the prices into the bar
        tick = float(msg["ask"]) if self.p.useask else float(msg["bid"])
        self.lines.open[0] = tick
        self.lines.high[0] = tick
        self.lines.low[0] = tick
        self.lines.close[0] = tick
        self.lines.volume[0] = 0.0
        self.lines.openinterest[0] = 0.0

        return True

    def _load_agg(self, msg):
        dtobj = pd.Timestamp(msg["time"], unit="ns")
        dt = date2num(dtobj)
        if dt <= self.lines.datetime[-1]:
            return False  # time already seen
        self.lines.datetime[0] = dt
        self.lines.open[0] = msg["open"]
        self.lines.high[0] = msg["high"]
        self.lines.low[0] = msg["low"]
        self.lines.close[0] = msg["close"]
        self.lines.volume[0] = msg["volume"]
        self.lines.openinterest[0] = 0.0

        return True

    def _load_history(self, msg):
        dtobj = msg["time"].to_pydatetime()
        dt = date2num(dtobj)
        if dt <= self.lines.datetime[-1]:
            return False  # time already seen

        # Common fields
        self.lines.datetime[0] = dt
        self.lines.volume[0] = msg["volume"]
        self.lines.openinterest[0] = 0.0

        # Put the prices into the bar
        self.lines.open[0] = msg["open"]
        self.lines.high[0] = msg["high"]
        self.lines.low[0] = msg["low"]
        self.lines.close[0] = msg["close"]

        return True
