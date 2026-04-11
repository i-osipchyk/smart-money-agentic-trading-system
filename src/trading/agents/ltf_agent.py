from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from trading.core.models import MarketState, PointOfInterest, TradeDecision, Trend
from trading.signals import detect_bos, detect_fractals, detect_fvg

llm = ChatAnthropic(model_name="claude-opus-4-5")  # type: ignore[call-arg]

LTF_SYSTEM_PROMPT = """You are an expert Smart Money Concepts trader analyzing lower timeframe price action.

You will receive:
- The higher timeframe trend and analysis
- Higher timeframe Points of Interest (POIs)
- Lower timeframe fractals, BOS events, and FVGs

Your job is to:
1. Determine if price is near a significant HTF POI
2. Look for LTF confirmation signals (BOS in direction of HTF trend, FVG as entry)
3. If confirmed, define entry, stop loss, and take profit levels
4. If not confirmed, explain why you are skipping this setup

Respond in this exact format:
SHOULD_TRADE: <yes|no>
DIRECTION: <bullish|bearish|none>
ENTRY: <price or none>
STOP_LOSS: <price or none>
TAKE_PROFIT: <price or none>
CONFIDENCE: <high|medium|low>
REASONING: <your reasoning in 2-3 sentences>"""


def _format_ltf_context(state: MarketState) -> str:
    lines = []

    lines.append("=== HTF CONTEXT ===")
    lines.append(f"Trend: {state.trend.value if state.trend else 'unknown'}")
    lines.append(f"Analysis: {state.htf_analysis or 'none'}")

    lines.append("\n=== HTF POINTS OF INTEREST ===")
    for poi in state.points_of_interest:
        lines.append(
            f"{poi.signal_type.value.upper()} | {poi.trend.value.upper()} | "
            f"zone: {poi.price_bottom:.2f} - {poi.price_top:.2f} | {poi.description}"
        )

    ltf_fractals = detect_fractals(state.ltf_candles, state.ltf_timeframe)
    ltf_bos = detect_bos(state.ltf_candles, ltf_fractals, state.ltf_timeframe)
    ltf_fvgs = detect_fvg(state.ltf_candles, state.ltf_timeframe)

    lines.append("\n=== LTF FRACTALS ===")
    for f in ltf_fractals[-10:]:
        kind = "HIGH" if f.is_high else "LOW"
        lines.append(f"{f.timestamp.date()} | {kind} | price: {f.price:.2f}")

    lines.append("\n=== LTF BREAK OF STRUCTURE ===")
    for b in ltf_bos[-5:]:
        lines.append(f"{b.timestamp.date()} | {b.trend.value.upper()} BOS | level: {b.level:.2f}")

    lines.append("\n=== LTF FAIR VALUE GAPS ===")
    for fvg in ltf_fvgs[-5:]:
        lines.append(
            f"{fvg.timestamp.date()} | {fvg.trend.value.upper()} FVG | "
            f"top: {fvg.top:.2f} | bottom: {fvg.bottom:.2f}"
        )

    current_price = state.ltf_candles["close"].iloc[-1]
    lines.append(f"\n=== CURRENT PRICE ===\n{current_price:.2f}")

    return "\n".join(lines)


def _parse_response(response: str, state: MarketState) -> TradeDecision:
    lines = response.strip().split("\n")
    should_trade = False
    direction = None
    entry = None
    stop_loss = None
    take_profit = None
    confidence = "low"
    reasoning = ""

    for line in lines:
        if line.startswith("SHOULD_TRADE:"):
            should_trade = line.split(":", 1)[1].strip().lower() == "yes"
        elif line.startswith("DIRECTION:"):
            val = line.split(":", 1)[1].strip().lower()
            direction = Trend(val) if val in Trend._value2member_map_ else None
        elif line.startswith("ENTRY:"):
            val = line.split(":", 1)[1].strip()
            entry = float(val) if val.lower() != "none" else None
        elif line.startswith("STOP_LOSS:"):
            val = line.split(":", 1)[1].strip()
            stop_loss = float(val) if val.lower() != "none" else None
        elif line.startswith("TAKE_PROFIT:"):
            val = line.split(":", 1)[1].strip()
            take_profit = float(val) if val.lower() != "none" else None
        elif line.startswith("CONFIDENCE:"):
            confidence = line.split(":", 1)[1].strip().lower()
        elif line.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return TradeDecision(
        symbol=state.symbol,
        should_trade=should_trade,
        direction=direction,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasoning=reasoning,
        confidence=confidence,
    )


def run_ltf_agent(state: MarketState) -> MarketState:
    context = _format_ltf_context(state)

    messages = [
        SystemMessage(content=LTF_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Symbol: {state.symbol}\nLTF: {state.ltf_timeframe.value}\n\n{context}"
        ),
    ]

    response = llm.invoke(messages)
    response_text = str(response.content)

    decision = _parse_response(response_text, state)

    return MarketState(
        **{
            **state.model_dump(exclude={"htf_candles", "ltf_candles"}),
            "htf_candles": state.htf_candles,
            "ltf_candles": state.ltf_candles,
            "trade_decision": decision.model_dump_json(indent=2),
        }
    )
