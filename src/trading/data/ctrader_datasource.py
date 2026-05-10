"""CTrader Open API data source for CFDs, commodities, and currencies."""

from __future__ import annotations

import asyncio
import ssl
import struct
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import (
    ProtoErrorRes,
    ProtoMessage,
)
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAAccountAuthReq,
    ProtoOAAccountAuthRes,
    ProtoOAApplicationAuthReq,
    ProtoOAApplicationAuthRes,
    ProtoOAErrorRes,
    ProtoOAGetTrendbarsReq,
    ProtoOAGetTrendbarsRes,
    ProtoOASymbolByIdReq,
    ProtoOASymbolByIdRes,
    ProtoOASymbolsListReq,
    ProtoOASymbolsListRes,
)
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATrendbarPeriod

_TIMEFRAME_TO_PERIOD: dict[str, int] = {
    "1m": ProtoOATrendbarPeriod.Value("M1"),
    "5m": ProtoOATrendbarPeriod.Value("M5"),
    "15m": ProtoOATrendbarPeriod.Value("M15"),
    "30m": ProtoOATrendbarPeriod.Value("M30"),
    "1h": ProtoOATrendbarPeriod.Value("H1"),
    "4h": ProtoOATrendbarPeriod.Value("H4"),
    "12h": ProtoOATrendbarPeriod.Value("H12"),
    "1d": ProtoOATrendbarPeriod.Value("D1"),
    "1w": ProtoOATrendbarPeriod.Value("W1"),
}

_PERIOD_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "12h": 720,
    "1d": 1440,
    "1w": 10080,
}

_PROTO_ERROR_PAYLOAD_TYPE = ProtoErrorRes().payloadType  # 50
_OA_ERROR_PAYLOAD_TYPE = ProtoOAErrorRes().payloadType  # 2142
_MAX_BARS_PER_PAGE = 4800


class CTraderDataSource:
    """DataSource for CFDs, commodities, and currencies via cTrader Open API.

    Symbols use cTrader naming (e.g. "EURUSD", "XAUUSD", "BTCUSD", "US500").
    Credentials come from cTrader Open API application settings.
    """

    _DEFAULT_HOST = "live.ctraderapi.com"
    _DEFAULT_PORT = 5035

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        access_token: str,
        account_id: int,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._access_token = access_token
        self._account_id = account_id
        self._host = host
        self._port = port
        # Cache: symbol_name → (symbol_id, digits)
        self._symbol_cache: dict[str, tuple[int, int]] = {}

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        until: datetime | None = None,
    ) -> pd.DataFrame:
        if timeframe not in _TIMEFRAME_TO_PERIOD:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported: {sorted(_TIMEFRAME_TO_PERIOD)}"
            )
        return asyncio.run(self._fetch(symbol, timeframe, limit, until))

    def fetch_range(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime,
        log: Callable[[str], None] | None = None,
    ) -> pd.DataFrame:
        """Fetch all candles in [since, until] with pagination. Used for backtesting."""
        since_ms = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000)
        return asyncio.run(
            self._fetch_range(symbol, timeframe, since_ms, until_ms, log)
        )

    async def _fetch(
        self, symbol: str, timeframe: str, limit: int, until: datetime | None = None
    ) -> pd.DataFrame:
        minutes = _PERIOD_MINUTES[timeframe]
        period_ms = minutes * 60 * 1000
        # Subtract one period so toTimestamp is the close time of the last bar,
        # matching Binance's convention where `until` = close time of last candle.
        to_ts = int((until or datetime.now(UTC)).timestamp() * 1000) - period_ms
        from_ts = to_ts - period_ms * limit

        ssl_ctx = ssl.create_default_context()
        reader, writer = await asyncio.open_connection(
            self._host, self._port, ssl=ssl_ctx
        )
        try:
            await self._app_auth(reader, writer)
            await self._account_auth(reader, writer)
            symbol_id, digits = await self._resolve_symbol(reader, writer, symbol)
            trendbars = await self._get_trendbars(
                reader, writer, symbol_id, timeframe, from_ts, to_ts, limit
            )
        finally:
            writer.close()
            await writer.wait_closed()

        return _decode_trendbars(trendbars, digits, limit)

    async def _app_auth(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        req = ProtoOAApplicationAuthReq()
        req.clientId = self._client_id
        req.clientSecret = self._client_secret
        await _send(writer, req)
        await _recv_type(reader, ProtoOAApplicationAuthRes)

    async def _account_auth(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = self._account_id
        req.accessToken = self._access_token
        await _send(writer, req)
        await _recv_type(reader, ProtoOAAccountAuthRes)

    async def _resolve_symbol(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        symbol: str,
    ) -> tuple[int, int]:
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        list_req = ProtoOASymbolsListReq()
        list_req.ctidTraderAccountId = self._account_id
        list_req.includeArchivedSymbols = False
        await _send(writer, list_req)
        list_res: ProtoOASymbolsListRes = await _recv_type(reader, ProtoOASymbolsListRes)

        symbol_id: int | None = None
        for s in list_res.symbol:
            if s.symbolName == symbol:
                symbol_id = s.symbolId
                break

        if symbol_id is None:
            raise ValueError(f"Symbol '{symbol}' not found on this cTrader account")

        id_req = ProtoOASymbolByIdReq()
        id_req.ctidTraderAccountId = self._account_id
        id_req.symbolId.append(symbol_id)
        await _send(writer, id_req)
        id_res: ProtoOASymbolByIdRes = await _recv_type(reader, ProtoOASymbolByIdRes)

        digits = id_res.symbol[0].digits
        self._symbol_cache[symbol] = (symbol_id, digits)
        return symbol_id, digits

    async def _get_trendbars(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        symbol_id: int,
        timeframe: str,
        from_ts: int,
        to_ts: int,
        limit: int,
    ) -> list[Any]:
        req = ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = self._account_id
        req.symbolId = symbol_id
        req.period = _TIMEFRAME_TO_PERIOD[timeframe]
        req.fromTimestamp = from_ts
        req.toTimestamp = to_ts
        req.count = limit
        await _send(writer, req)
        res: ProtoOAGetTrendbarsRes = await _recv_type(reader, ProtoOAGetTrendbarsRes)
        return list(res.trendbar)


    async def _paginate(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        symbol_id: int,
        timeframe: str,
        since_ms: int,
        until_ms: int,
        log: Callable[[str], None] | None = None,
        label: str = "",
    ) -> list[Any]:
        """Fetch all trendbar pages for a range on an already-open connection."""
        period_ms = _PERIOD_MINUTES[timeframe] * 60 * 1000
        prefix = f"[{label}] " if label else ""
        all_bars: list[Any] = []
        current_from = since_ms
        last_seen_bar_ts_ms = -1
        page = 0

        while current_from < until_ms:
            page += 1
            if log is not None:
                ts = datetime.fromtimestamp(current_from / 1000, tz=UTC)
                log(f"  {prefix}page {page} (since {ts.strftime('%Y-%m-%d %H:%M')} UTC) …")

            bars = await self._get_trendbars(
                reader, writer, symbol_id, timeframe,
                current_from, until_ms, _MAX_BARS_PER_PAGE,
            )
            if not bars:
                break
            all_bars.extend(bars)

            last_bar_ts_ms = bars[-1].utcTimestampInMinutes * 60 * 1000
            if log is not None:
                log(f"  {prefix}got {len(bars)} bars")

            # API returns same trailing bars when no data exists past current_from
            if last_bar_ts_ms <= last_seen_bar_ts_ms:
                break
            last_seen_bar_ts_ms = last_bar_ts_ms

            if len(bars) < _MAX_BARS_PER_PAGE:
                break
            current_from = last_bar_ts_ms + period_ms

        return all_bars

    async def _fetch_range(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        until_ms: int,
        log: Callable[[str], None] | None = None,
    ) -> pd.DataFrame:
        ssl_ctx = ssl.create_default_context()
        reader, writer = await asyncio.open_connection(
            self._host, self._port, ssl=ssl_ctx
        )
        try:
            if log is not None:
                log("Connecting to cTrader …")
            await self._app_auth(reader, writer)
            if log is not None:
                log("App authenticated")
            await self._account_auth(reader, writer)
            if log is not None:
                log("Account authenticated")
            symbol_id, digits = await self._resolve_symbol(reader, writer, symbol)
            if log is not None:
                log(f"Symbol resolved: {symbol} (id={symbol_id}, digits={digits})")
            all_bars = await self._paginate(
                reader, writer, symbol_id, timeframe, since_ms, until_ms, log
            )
        finally:
            writer.close()
            await writer.wait_closed()

        df = _decode_trendbars(all_bars, digits, len(all_bars))
        since_dt = datetime.fromtimestamp(since_ms / 1000, tz=UTC)
        until_dt = datetime.fromtimestamp(until_ms / 1000, tz=UTC)
        mask = (df["timestamp"] >= since_dt) & (df["timestamp"] <= until_dt)
        return df[mask].reset_index(drop=True)

    def fetch_both_ranges(
        self,
        symbol: str,
        htf_timeframe: str,
        htf_since: datetime,
        ltf_timeframe: str,
        ltf_since: datetime,
        until: datetime,
        log: Callable[[str], None] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch HTF and LTF ranges in a single authenticated connection."""
        return asyncio.run(
            self._fetch_both_ranges(
                symbol,
                htf_timeframe, int(htf_since.timestamp() * 1000),
                ltf_timeframe, int(ltf_since.timestamp() * 1000),
                int(until.timestamp() * 1000),
                log,
            )
        )

    async def _fetch_both_ranges(
        self,
        symbol: str,
        htf_tf: str,
        htf_since_ms: int,
        ltf_tf: str,
        ltf_since_ms: int,
        until_ms: int,
        log: Callable[[str], None] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ssl_ctx = ssl.create_default_context()
        reader, writer = await asyncio.open_connection(
            self._host, self._port, ssl=ssl_ctx
        )
        try:
            if log is not None:
                log("Connecting to cTrader …")
            await self._app_auth(reader, writer)
            if log is not None:
                log("App authenticated")
            await self._account_auth(reader, writer)
            if log is not None:
                log("Account authenticated")
            symbol_id, digits = await self._resolve_symbol(reader, writer, symbol)
            if log is not None:
                log(f"Symbol resolved: {symbol} (id={symbol_id}, digits={digits})")
            if log is not None:
                log(f"Fetching {htf_tf} bars …")
            htf_bars = await self._paginate(
                reader, writer, symbol_id, htf_tf, htf_since_ms, until_ms, log,
                label=htf_tf,
            )
            if log is not None:
                log(f"Fetching {ltf_tf} bars …")
            ltf_bars = await self._paginate(
                reader, writer, symbol_id, ltf_tf, ltf_since_ms, until_ms, log,
                label=ltf_tf,
            )
        finally:
            writer.close()
            await writer.wait_closed()

        def _as_df(bars: list[Any], since_ms: int) -> pd.DataFrame:
            df = _decode_trendbars(bars, digits, len(bars))
            since_dt = datetime.fromtimestamp(since_ms / 1000, tz=UTC)
            until_dt = datetime.fromtimestamp(until_ms / 1000, tz=UTC)
            mask = (df["timestamp"] >= since_dt) & (df["timestamp"] <= until_dt)
            return df[mask].reset_index(drop=True)

        return _as_df(htf_bars, htf_since_ms), _as_df(ltf_bars, ltf_since_ms)


async def _send(writer: asyncio.StreamWriter, inner_msg: Any) -> None:
    outer = ProtoMessage()
    outer.payloadType = inner_msg.payloadType
    outer.payload = inner_msg.SerializeToString()
    data = outer.SerializeToString()
    writer.write(struct.pack(">I", len(data)) + data)
    await writer.drain()
    await asyncio.sleep(0.4)  # stay under cTrader rate limit


async def _recv_type(
    reader: asyncio.StreamReader, expected_type: type[Any]
) -> Any:
    expected_payload_type = expected_type().payloadType
    while True:
        length_bytes = await reader.readexactly(4)
        length = struct.unpack(">I", length_bytes)[0]
        data = await reader.readexactly(length)

        outer = ProtoMessage()
        outer.ParseFromString(data)

        if outer.payloadType == expected_payload_type:
            inner = expected_type()
            inner.ParseFromString(outer.payload)
            return inner

        if outer.payloadType == _PROTO_ERROR_PAYLOAD_TYPE:
            err = ProtoErrorRes()
            err.ParseFromString(outer.payload)
            raise RuntimeError(f"cTrader protocol error {err.errorCode}: {err.description}")

        if outer.payloadType == _OA_ERROR_PAYLOAD_TYPE:
            err = ProtoOAErrorRes()
            err.ParseFromString(outer.payload)
            raise RuntimeError(f"cTrader OA error {err.errorCode}: {err.description}")


_PRICE_DIVISOR = 100_000  # cTrader encodes all trendbar prices as integer * 10^-5


def _decode_trendbars(trendbars: list[Any], digits: int, limit: int) -> pd.DataFrame:
    """Convert raw ProtoOATrendbar objects to a standard OHLCV DataFrame."""
    divisor = _PRICE_DIVISOR
    rows = []
    for bar in trendbars:
        low = bar.low / divisor
        rows.append({
            "timestamp": datetime.fromtimestamp(bar.utcTimestampInMinutes * 60, tz=UTC),
            "open": (bar.low + bar.deltaOpen) / divisor,
            "high": (bar.low + bar.deltaHigh) / divisor,
            "low": low,
            "close": (bar.low + bar.deltaClose) / divisor,
            "volume": float(bar.volume),
        })

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df.tail(limit).reset_index(drop=True)
