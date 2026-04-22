"""CTrader Open API data source for CFDs, commodities, and currencies."""

from __future__ import annotations

import asyncio
import ssl
import struct
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
    ) -> pd.DataFrame:
        if timeframe not in _TIMEFRAME_TO_PERIOD:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported: {sorted(_TIMEFRAME_TO_PERIOD)}"
            )
        return asyncio.run(self._fetch(symbol, timeframe, limit))

    async def _fetch(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        minutes = _PERIOD_MINUTES[timeframe]
        to_ts = int(datetime.now(UTC).timestamp() * 1000)
        from_ts = to_ts - minutes * 60 * 1000 * limit

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


async def _send(writer: asyncio.StreamWriter, inner_msg: Any) -> None:
    outer = ProtoMessage()
    outer.payloadType = inner_msg.payloadType
    outer.payload = inner_msg.SerializeToString()
    data = outer.SerializeToString()
    writer.write(struct.pack(">I", len(data)) + data)
    await writer.drain()


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


def _decode_trendbars(trendbars: list[Any], digits: int, limit: int) -> pd.DataFrame:
    """Convert raw ProtoOATrendbar objects to a standard OHLCV DataFrame."""
    divisor = 10**digits
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
