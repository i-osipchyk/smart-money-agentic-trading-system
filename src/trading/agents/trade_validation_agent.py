from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from trading.agents.llm_provider import DEFAULT_CONFIG, LLMConfig, create_llm_client
from trading.core.models import StrategySetup, TradeDecision

load_dotenv()

# Sentinel string that separates free-form reasoning from the parseable block.
_DECISION_FENCE = "```decision"


def build_prompt(setup: StrategySetup) -> str:
    """
    Build the prompt that will be fed to the trade validation agent.

    Assembles all fields of StrategySetup into a structured prompt and
    appends a required structured-output block so ``parse_decision`` can
    extract a machine-readable ``TradeDecision``.

    The agent's job is to select the best take-profit target from the
    provided candidate list.  Entry and stop-loss are fixed by the strategy;
    the final limit-order entry to achieve 2:1 RR is computed by
    ``parse_decision`` after the agent picks a TP.

    Args:
        setup: StrategySetup produced by a strategy's detect_entry().

    Returns:
        Prompt string ready to be sent to an LLM.
    """
    return (
        "You are a professional cryptocurrency trader specializing in "
        "Smart Money Concepts (SMC).\n"
        "\n"
        "## Input Data\n"
        f"{setup.input_data}\n"
        "\n"
        "## Strategy\n"
        f"{setup.strategy_description}\n"
        "\n"
        f"## Detected Setup {setup.direction.value.upper()}\n"
        "\n"
        "### HTF Point of Interest\n"
        f"{setup.htf_poi}\n"
        "\n"
        "### LTF Confirmation\n"
        f"{setup.confirm_details}\n"
        "\n"
        f"### BOS Level (natural entry): {setup.entry:,.2f}\n"
        "\n"
        f"### Stop Loss: {setup.stop_loss:,.2f}\n"
        "\n"
        "### Potential Targets\n"
        f"{setup.target}\n"
        "\n"
        f"## {setup.candles}\n"
        "\n"
        "## Task\n"
        "Analyze the detected setup in the context of the market structure above.\n"
        "Select the single best take-profit target from the candidate list that\n"
        "aligns with the current market structure. Consider liquidity levels, key\n"
        "HTF zones, and whether price is likely to reach the target before reversing.\n"
        "Give the priority to most recent and the most important points, that were global swings.\n"
        "If no candidate is a valid target given the current structure, output NO.\n"
        "\n"
        "Note: the stop loss is fixed. The system will place a limit order between\n"
        "the BOS level and the stop loss to achieve 2:1 RR with your chosen target.\n"
        "If your chosen target gives less than 1:1 RR at the BOS level, no trade\n"
        "is taken.\n"
        "\n"
        "## Required Response Format\n"
        "After your analysis, end your response with **exactly** this block "
        "(replace the values, keep the keys verbatim):\n"
        "\n"
        "```decision\n"
        "should_trade: YES or NO\n"
        "target_price: <number from the candidate list, or omit if NO>\n"
        "confidence: HIGH, MEDIUM, or LOW\n"
        "reasoning: one-sentence summary of the decisive factor\n"
        "```"
    )


def parse_decision(symbol: str, response: str, setup: StrategySetup) -> TradeDecision:
    """
    Parse the agent's raw response into a ``TradeDecision``.

    Looks for the structured ``decision`` code block appended by the prompt.
    Falls back to ``should_trade=False`` when the block is absent — without a
    chosen ``target_price`` there is no way to compute levels.

    Level computation (when agent says YES and provides a target_price):

    1. TP = agent's target_price.
    2. SL = setup.stop_loss (fixed by strategy).
    3. BOS = setup.entry (natural entry level).
    4. Check RR at BOS:  if reward-at-BOS / risk < 1  →  no trade.
    5. Compute the limit-order entry that gives exactly 2:1 RR:
         bullish:  entry = (TP + 2 × SL) / 3  capped at BOS (can't enter above BOS)
         bearish:  entry = (2 × SL + TP) / 3  capped at BOS (can't enter below BOS)

    Args:
        symbol:   Trading pair.
        response: Raw text returned by the agent.
        setup:    The StrategySetup that was passed to build_prompt().

    Returns:
        A TradeDecision ready for order simulation.
    """
    should_trade = False
    target_price: float | None = None
    confidence = "n/a"
    reasoning = response.strip()

    if _DECISION_FENCE in response:
        try:
            block_start = response.index(_DECISION_FENCE) + len(_DECISION_FENCE)
            block_end = response.index("```", block_start)
            block = response[block_start:block_end].strip()
            for line in block.splitlines():
                key, _, val = line.partition(":")
                key = key.strip().lower()
                val = val.strip()
                if key == "should_trade":
                    should_trade = val.upper() == "YES"
                elif key == "target_price":
                    try:
                        target_price = float(val.replace(",", ""))
                    except ValueError:
                        pass
                elif key == "confidence":
                    confidence = val.upper()
                elif key == "reasoning":
                    reasoning = val
        except (ValueError, IndexError):
            pass  # structured block malformed — fall through to no-trade

    # Agent said YES but gave no parseable target → can't compute levels
    if should_trade and target_price is None:
        should_trade = False
        reasoning = f"[no parseable target_price] {reasoning}"

    if not should_trade:
        return TradeDecision(
            symbol=symbol,
            should_trade=False,
            reasoning=reasoning,
            confidence=confidence,
        )

    # --- level computation ---------------------------------------------------
    assert target_price is not None  # guaranteed by the early returns above
    bos = setup.entry
    sl = setup.stop_loss
    tp: float = target_price
    bullish = setup.direction.value == "bullish"

    reward_at_bos = (tp - bos) if bullish else (bos - tp)
    risk_at_bos = (bos - sl) if bullish else (sl - bos)

    if reward_at_bos < risk_at_bos:
        # RR at BOS is less than 1:1 — no trade
        return TradeDecision(
            symbol=symbol,
            should_trade=False,
            reasoning=(
                f"RR at BOS < 1:1 "
                f"(reward {reward_at_bos:,.2f} / risk {risk_at_bos:,.2f})"
            ),
            confidence=confidence,
        )

    # Limit-order entry that gives exactly 2:1 with the chosen TP and SL.
    # For bullish: entry = (TP + 2×SL) / 3  →  moves down from BOS toward SL.
    # For bearish: entry = (2×SL + TP) / 3  →  moves up from BOS toward SL.
    if bullish:
        entry = (tp + 2 * sl) / 3
    else:
        entry = (2 * sl + tp) / 3

    # If the computed entry is already through the BOS candle's close, the limit
    # would fill immediately — treat it as a market order at next candle open.
    bos_close = setup.bos_candle_close
    is_market_order = (bullish and entry >= bos_close) or (
        not bullish and entry <= bos_close
    )

    return TradeDecision(
        symbol=symbol,
        should_trade=True,
        direction=setup.direction,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        is_market_order=is_market_order,
        reasoning=reasoning,
        confidence=confidence,
    )


class TradeValidationAgent:
    def __init__(self, config: LLMConfig = DEFAULT_CONFIG) -> None:
        self._llm: BaseChatModel = create_llm_client(config)

    def run(self, prompt: str) -> str:
        """
        Send the prompt to Claude and return the analysis response.

        Args:
            prompt: Full prompt string produced by build_prompt().

        Returns:
            Raw response text from the model.
        """
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return str(response.content)
