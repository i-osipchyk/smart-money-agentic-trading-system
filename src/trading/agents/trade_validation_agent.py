from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from trading.core.models import StrategySetup, TradeDecision

load_dotenv()

_MODEL = "claude-opus-4-5"

# Sentinel string that separates free-form reasoning from the parseable block.
_DECISION_FENCE = "```decision"


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
        f"## {setup.candles}\n"
        "\n"
        "## Task\n"
        "Analyze the detected setup in the context of the HTF candle data above.\n"
        "Determine whether this is a valid, high-probability trade entry.\n"
        "Provide your reasoning and a clear decision.\n"
        "\n"
        "## Required Response Format\n"
        "After your analysis, end your response with **exactly** this block "
        "(replace the values, keep the keys verbatim):\n"
        "\n"
        "```decision\n"
        "should_trade: YES or NO\n"
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

    return TradeDecision(
        symbol=symbol,
        should_trade=should_trade,
        direction=setup.direction if should_trade else None,
        entry_price=setup.entry if should_trade else None,
        stop_loss=setup.stop_loss if should_trade else None,
        take_profit=setup.take_profit if should_trade else None,
        reasoning=reasoning,
        confidence=confidence,
    )


class TradeValidationAgent:
    def __init__(self, model: str = _MODEL) -> None:
        self._llm = ChatAnthropic(model_name=model)  # type: ignore[call-arg]

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
