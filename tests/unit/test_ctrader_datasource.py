"""Unit tests for CTraderDataSource."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from trading.data.ctrader_datasource import CTraderDataSource, _decode_trendbars


def _make_trendbar(
    low_abs: int,
    delta_open: int,
    delta_high: int,
    delta_close: int,
    volume: int,
    ts_minutes: int,
) -> MagicMock:
    bar = MagicMock()
    bar.low = low_abs
    bar.deltaOpen = delta_open
    bar.deltaHigh = delta_high
    bar.deltaClose = delta_close
    bar.volume = volume
    bar.utcTimestampInMinutes = ts_minutes
    return bar


# ---------------------------------------------------------------------------
# _decode_trendbars — pure transformation, no I/O
# ---------------------------------------------------------------------------


class TestDecodeTrendbars:
    def test_ohlcv_columns_present(self) -> None:
        bar = _make_trendbar(105000, 100, 300, 50, 1000, 28000000)
        df = _decode_trendbars([bar], digits=5, limit=1)
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_price_decoding_five_digits(self) -> None:
        # EURUSD: low=1.05000, open=1.05100, high=1.05300, close=1.05050
        bar = _make_trendbar(105000, 100, 300, 50, 1000, 28000000)
        df = _decode_trendbars([bar], digits=5, limit=1)
        row = df.iloc[0]
        assert abs(row["low"] - 1.05000) < 1e-9
        assert abs(row["open"] - 1.05100) < 1e-9
        assert abs(row["high"] - 1.05300) < 1e-9
        assert abs(row["close"] - 1.05050) < 1e-9

    def test_price_decoding_two_digits(self) -> None:
        # XAUUSD: low=1800.00, high=1850.00
        bar = _make_trendbar(180000, 0, 5000, 2500, 50, 1000)
        df = _decode_trendbars([bar], digits=2, limit=1)
        assert abs(df.iloc[0]["low"] - 1800.00) < 1e-8
        assert abs(df.iloc[0]["high"] - 1850.00) < 1e-8

    def test_volume_converted_to_float(self) -> None:
        bar = _make_trendbar(100000, 0, 0, 0, 500, 1000)
        df = _decode_trendbars([bar], digits=5, limit=1)
        assert isinstance(df.iloc[0]["volume"], float)
        assert df.iloc[0]["volume"] == 500.0

    def test_timestamp_epoch_zero(self) -> None:
        bar = _make_trendbar(100000, 0, 0, 0, 0, ts_minutes=0)
        df = _decode_trendbars([bar], digits=5, limit=1)
        assert df.iloc[0]["timestamp"] == datetime(1970, 1, 1, tzinfo=UTC)

    def test_timestamp_is_utc_aware(self) -> None:
        bar = _make_trendbar(100000, 0, 0, 0, 0, ts_minutes=1000)
        df = _decode_trendbars([bar], digits=5, limit=1)
        assert df.iloc[0]["timestamp"].tzinfo is not None

    def test_timestamp_from_minutes(self) -> None:
        # 60 minutes from epoch = 1970-01-01 01:00 UTC
        bar = _make_trendbar(100000, 0, 0, 0, 0, ts_minutes=60)
        df = _decode_trendbars([bar], digits=5, limit=1)
        assert df.iloc[0]["timestamp"] == datetime(1970, 1, 1, 1, 0, tzinfo=UTC)

    def test_limit_truncates_to_last_n(self) -> None:
        bars = [_make_trendbar(100000, 0, 0, 0, i, ts_minutes=i) for i in range(10)]
        df = _decode_trendbars(bars, digits=5, limit=5)
        assert len(df) == 5
        # tail(5) keeps the last 5 (volumes 5–9)
        assert list(df["volume"]) == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_limit_larger_than_bars_returns_all(self) -> None:
        bars = [_make_trendbar(100000, 0, 0, 0, i, ts_minutes=i) for i in range(3)]
        df = _decode_trendbars(bars, digits=5, limit=100)
        assert len(df) == 3

    def test_index_reset_after_tail(self) -> None:
        bars = [_make_trendbar(100000, 0, 0, 0, i, ts_minutes=i) for i in range(5)]
        df = _decode_trendbars(bars, digits=5, limit=3)
        assert list(df.index) == [0, 1, 2]

    def test_empty_bars_returns_empty_dataframe(self) -> None:
        df = _decode_trendbars([], digits=5, limit=10)
        assert df.empty
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_multiple_bars_ordered_correctly(self) -> None:
        bars = [
            _make_trendbar(100000, 100, 200, 50, 100, ts_minutes=0),
            _make_trendbar(101000, 200, 400, 300, 200, ts_minutes=60),
        ]
        df = _decode_trendbars(bars, digits=5, limit=2)
        assert len(df) == 2
        assert df.iloc[0]["timestamp"] < df.iloc[1]["timestamp"]


# ---------------------------------------------------------------------------
# CTraderDataSource.get_ohlcv — interface contract
# ---------------------------------------------------------------------------


class TestGetOhlcvValidation:
    def test_unsupported_timeframe_raises_value_error(self) -> None:
        source = CTraderDataSource("id", "secret", "token", 12345)
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            source.get_ohlcv("EURUSD", "3h", 10)

    def test_all_supported_timeframes_do_not_raise_on_validation(self) -> None:
        source = CTraderDataSource("id", "secret", "token", 12345)
        supported = ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d", "1w"]
        expected_df = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        import inspect

        def close_and_return(coro: object) -> pd.DataFrame:
            if inspect.iscoroutine(coro):
                coro.close()  # type: ignore[union-attr]
            return expected_df

        for tf in supported:
            with patch(
                "trading.data.ctrader_datasource.asyncio.run",
                side_effect=close_and_return,
            ):
                result = source.get_ohlcv("EURUSD", tf, 10)
                assert isinstance(result, pd.DataFrame)

    def test_get_ohlcv_returns_dataframe_from_fetch(self) -> None:
        source = CTraderDataSource("id", "secret", "token", 12345)
        expected_df = pd.DataFrame({
            "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
            "open": [1.10],
            "high": [1.11],
            "low": [1.09],
            "close": [1.105],
            "volume": [5000.0],
        })

        def close_and_return(coro: object) -> pd.DataFrame:
            import inspect
            if inspect.iscoroutine(coro):
                coro.close()  # type: ignore[union-attr]
            return expected_df

        with patch(
            "trading.data.ctrader_datasource.asyncio.run",
            side_effect=close_and_return,
        ) as mock_run:
            result = source.get_ohlcv("EURUSD", "1h", 1)
        assert result is expected_df
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# CTraderDataSource._resolve_symbol — symbol cache and not-found error
# ---------------------------------------------------------------------------


class TestResolveSymbol:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_network(self) -> None:
        source = CTraderDataSource("id", "secret", "token", 12345)
        source._symbol_cache["EURUSD"] = (42, 5)

        reader = AsyncMock()
        writer = AsyncMock()

        result = await source._resolve_symbol(reader, writer, "EURUSD")

        assert result == (42, 5)
        reader.readexactly.assert_not_called()

    @pytest.mark.asyncio
    async def test_symbol_not_found_raises_value_error(self) -> None:
        source = CTraderDataSource("id", "secret", "token", 12345)

        # Build a fake SymbolsListRes with one symbol that doesn't match
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOASymbolsListRes
        from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOALightSymbol

        light = ProtoOALightSymbol()
        light.symbolId = 1
        light.symbolName = "GBPUSD"

        fake_res = ProtoOASymbolsListRes()
        fake_res.ctidTraderAccountId = 12345
        fake_res.symbol.append(light)

        with patch(
            "trading.data.ctrader_datasource._send", new_callable=AsyncMock
        ), patch(
            "trading.data.ctrader_datasource._recv_type",
            new_callable=AsyncMock,
            return_value=fake_res,
        ):
            with pytest.raises(ValueError, match="Symbol 'EURUSD' not found"):
                await source._resolve_symbol(AsyncMock(), AsyncMock(), "EURUSD")

    @pytest.mark.asyncio
    async def test_resolve_symbol_populates_cache(self) -> None:
        source = CTraderDataSource("id", "secret", "token", 12345)

        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOASymbolByIdRes,
            ProtoOASymbolsListRes,
        )
        from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
            ProtoOALightSymbol,
            ProtoOASymbol,
        )

        light = ProtoOALightSymbol()
        light.symbolId = 7
        light.symbolName = "EURUSD"

        list_res = ProtoOASymbolsListRes()
        list_res.ctidTraderAccountId = 12345
        list_res.symbol.append(light)

        full_sym = ProtoOASymbol()
        full_sym.symbolId = 7
        full_sym.digits = 5

        id_res = ProtoOASymbolByIdRes()
        id_res.ctidTraderAccountId = 12345
        id_res.symbol.append(full_sym)

        call_count = 0

        async def fake_recv_type(reader: object, expected_type: type) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return list_res
            return id_res

        with patch(
            "trading.data.ctrader_datasource._send", new_callable=AsyncMock
        ), patch(
            "trading.data.ctrader_datasource._recv_type", side_effect=fake_recv_type
        ):
            result = await source._resolve_symbol(AsyncMock(), AsyncMock(), "EURUSD")

        assert result == (7, 5)
        assert source._symbol_cache["EURUSD"] == (7, 5)
