from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from trading.core.models import FVG, BOS, Fractal, MarketState, PointOfInterest, SignalType, Timeframe, Trend
from trading.signals import detect_bos, detect_fractals, detect_fvg

llm = ChatAnthropic(model_name="claude-opus-4-5")  # type: ignore[call-arg]

HTF_SYSTEM_PROMPT = """You are an expert Smart Money Concepts trader analyzing higher timeframe market structure.

You will receive:
- Detected fractals (swing highs and lows)
- Break of Structure (BOS) events
- Fair Value Gaps (FVGs)

Your job is to:
1. Determine the overall market trend (bullish, bearish, or ranging)
2. Identify the most significant Points of Interest (POIs) where price may react
3. Explain your reasoning clearly

Respond in this exact format:
TREND: <bullish|bearish|ranging>
POI_COUNT: <number of significant POIs you identified>
POI_1: <price_top>|<price_bottom>|<bullish|bearish>|<fvg|bos|fractal>|<brief description>
POI_2: <price_top>|<price_bottom>|<bullish|bearish>|<fvg|bos|fractal>|<brief description>
... (continue for each POI)
ANALYSIS: <your overall analysis in 2-3 sentences>"""


def _format_signals(
    fractals: list[Fractal],
    bos_list: list[BOS],
    fvgs: list[FVG],
) -> str:
    lines = []

    lines.append("=== FRACTALS ===")
    for f in fractals[-10:]:
        kind = "HIGH" if f.is_high else "LOW"
        lines.append(f"{f.timestamp.date()} | {kind} | price: {f.price:.2f}")

    lines.append("\n=== BREAK OF STRUCTURE ===")
    for b in bos_list[-5:]:
        lines.append(f"{b.timestamp.date()} | {b.trend.value.upper()} BOS | level: {b.level:.2f}")

    lines.append("\n=== FAIR VALUE GAPS ===")
    for fvg in fvgs[-5:]:
        lines.append(
            f"{fvg.timestamp.date()} | {fvg.trend.value.upper()} FVG | "
            f"top: {fvg.top:.2f} | bottom: {fvg.bottom:.2f}"
        )

    return "\n".join(lines)


def _parse_response(response: str, timeframe: Timeframe) -> tuple[Trend, list[PointOfInterest], str]:
    lines = response.strip().split("\n")
    trend = Trend.RANGING
    pois: list[PointOfInterest] = []
    analysis = ""

    for line in lines:
        if line.startswith("TREND:"):
            trend_str = line.split(":", 1)[1].strip().lower()
            trend = Trend(trend_str) if trend_str in Trend._value2member_map_ else Trend.RANGING

        elif line.startswith("POI_") and "|" in line:
            try:
                content = line.split(":", 1)[1].strip()
                parts = content.split("|")
                if len(parts) >= 5:
                    pois.append(PointOfInterest(
                        timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                        price_top=float(parts[0].strip()),
                        price_bottom=float(parts[1].strip()),
                        trend=Trend(parts[2].strip()),
                        signal_type=SignalType(parts[3].strip()),
                        timeframe=timeframe,
                        description=parts[4].strip(),
                    ))
            except (ValueError, KeyError):
                continue

        elif line.startswith("ANALYSIS:"):
            analysis = line.split(":", 1)[1].strip()

    return trend, pois, analysis


def run_htf_agent(state: MarketState) -> MarketState:
    fractals = detect_fractals(state.htf_candles, state.htf_timeframe)
    bos_list = detect_bos(state.htf_candles, fractals, state.htf_timeframe)
    fvgs = detect_fvg(state.htf_candles, state.htf_timeframe)

    signal_summary = _format_signals(fractals, bos_list, fvgs)

    messages = [
        SystemMessage(content=HTF_SYSTEM_PROMPT),
        HumanMessage(content=f"Symbol: {state.symbol}\nTimeframe: {state.htf_timeframe.value}\n\n{signal_summary}"),
    ]

    response = llm.invoke(messages)
    response_text = str(response.content)

    trend, pois, analysis = _parse_response(response_text, state.htf_timeframe)

    return MarketState(
        **{
            **state.model_dump(exclude={"htf_candles", "ltf_candles"}),
            "htf_candles": state.htf_candles,
            "ltf_candles": state.ltf_candles,
            "trend": trend,
            "points_of_interest": pois,
            "fractals": fractals,
            "fvgs": fvgs,
            "bos_levels": bos_list,
            "htf_analysis": analysis,
        }
    )
