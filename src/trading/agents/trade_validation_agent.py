from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from trading.agents.llm_provider import DEFAULT_CONFIG, LLMConfig, create_llm_client
from trading.core.models import StrategySetup, TradeDecision

load_dotenv()

# Sentinel string that separates free-form reasoning from the parseable block.
_DECISION_FENCE = "```decision"


def _format_levels(setup: StrategySetup) -> str:
    entry = setup.entry
    sl = setup.stop_loss
    tp = setup.take_profit
    bullish = setup.direction.value == "bullish"
    risk = (entry - sl) if bullish else (sl - entry)
    reward = (tp - entry) if bullish else (entry - tp)
    rr = reward / risk if risk else 0.0
    risk_pct = risk / entry * 100 if entry else 0.0
    return "\n".join([
        f"Entry (limit):  {entry:,.2f}",
        f"Stop Loss:      {sl:,.2f}  (risk {risk:,.2f}, {risk_pct:.2f}%)",
        f"Take Profit:    {tp:,.2f}  ({rr:.1f}:1 RR)",
    ])


def build_prompt(setup: StrategySetup) -> str:
    """
    Build the prompt that will be fed to the trade validation agent.

    Assembles all fields of StrategySetup into a structured prompt and
    appends a required structured-output block so ``parse_decision`` can
    extract a machine-readable ``TradeDecision``.

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
        "### Potential Targets\n"
        f"{setup.target}\n"
        "\n"
        "### Computed Levels\n"
        f"{_format_levels(setup)}\n"
        "\n"
        f"## {setup.candles}\n"
        "\n"
        "## Task\n"
        "Evaluate the detected setup across these dimensions:\n"
        "1. HTF trend alignment (higher highs / higher lows visible in candle data?)\n"
        "2. Quality of the liquidity sweep (clean wick into FVG vs. close inside zone?)\n"
        "3. BOS candle strength (impulsive close or weak?)\n"
        "4. Target viability (is the TP a clean structural level or a wick extreme?)\n"
        "5. Risk/reward acceptability given the above"
        "\n"
        "## Required Response Format\n"
        "After your analysis, end your response with **exactly** this block "
        "(replace the values, keep the keys verbatim):\n"
        "\n"
        "Confidence rubric:\n"
        "HIGH:   all 5 dimensions align, clean structure\n"
        "MEDIUM: 3-4 dimensions align, minor concern present\n"
        "LOW:    significant structural ambiguity or conflict\n"
        "\n"
        "```decision\n"
        "should_trade: YES or NO\n"
        "entry: entry price as a number (omit if should_trade is NO)\n"
        "stop_loss: stop loss price as a number (omit if should_trade is NO)\n"
        "take_profit: take profit price as a number (omit if should_trade is NO)\n"
        "confidence: HIGH, MEDIUM, or LOW\n"
        "reasoning: one-sentence summary of the decisive factor\n"
        "```"
    )


def parse_decision(symbol: str, response: str, setup: StrategySetup) -> TradeDecision:
    """
    Parse the agent's raw response into a ``TradeDecision``.

    Looks for the structured ``decision`` code block appended by the prompt.
    Falls back to a keyword search on "NO TRADE" / "TRADE" if the block is
    absent, and uses the full response as the reasoning.

    Entry, stop-loss, and take-profit are always taken from the
    strategy-computed levels so that simulation results are comparable across
    setups regardless of what the agent suggests numerically.

    Args:
        symbol:   Trading pair.
        response: Raw text returned by the agent.
        setup:    The StrategySetup that was passed to build_prompt().

    Returns:
        A TradeDecision ready for order simulation.
    """
    should_trade = False
    confidence = "n/a"
    reasoning = response.strip()
    agent_entry: float | None = None
    agent_sl: float | None = None
    agent_tp: float | None = None
    agent_target: float | None = None  # backward-compat with old "target" key

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
                elif key == "entry":
                    try:
                        agent_entry = float(val.replace(",", ""))
                    except ValueError:
                        pass
                elif key == "stop_loss":
                    try:
                        agent_sl = float(val.replace(",", ""))
                    except ValueError:
                        pass
                elif key == "take_profit":
                    try:
                        agent_tp = float(val.replace(",", ""))
                    except ValueError:
                        pass
                elif key == "target":  # old format
                    try:
                        agent_target = float(val.replace(",", ""))
                    except ValueError:
                        pass
                elif key == "confidence":
                    confidence = val.upper()
                elif key == "reasoning":
                    reasoning = val
        except (ValueError, IndexError):
            pass  # fall through to keyword fallback
    else:
        # Fallback: crude keyword search
        upper = response.upper()
        should_trade = "NO TRADE" not in upper and "TRADE" in upper

    if not should_trade:
        return TradeDecision(
            symbol=symbol,
            should_trade=False,
            reasoning=reasoning,
            confidence=confidence,
        )

    entry = agent_entry if agent_entry is not None else setup.entry
    stop_loss = agent_sl if agent_sl is not None else setup.stop_loss
    if agent_tp is not None:
        take_profit = agent_tp
    elif agent_target is not None:
        take_profit = agent_target
    else:
        take_profit = setup.take_profit

    # Only adjust entry for 2:1 RR when the agent did not provide explicit entry/sl/tp.
    # Solving (|tp - e|) / (|e - sl|) = 2  →  e = (tp + 2*sl) / 3.
    if agent_entry is None and agent_sl is None and agent_tp is None:
        if setup.direction.value == "bullish":
            rr = (take_profit - entry) / (entry - stop_loss) if entry != stop_loss else 0
        else:
            rr = (entry - take_profit) / (stop_loss - entry) if stop_loss != entry else 0
        if rr < 2.0:
            entry = (take_profit + 2 * stop_loss) / 3

    return TradeDecision(
        symbol=symbol,
        should_trade=True,
        direction=setup.direction,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
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
