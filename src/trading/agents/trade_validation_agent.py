from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from trading.core.models import StrategySetup

load_dotenv()

_MODEL = "claude-opus-4-5"


def build_prompt(setup: StrategySetup) -> str:
    """
    Build the prompt that will be fed to the trade validation agent.

    Assembles all fields of StrategySetup into a structured prompt:
      - Input data (symbol, timeframes, ranges, params)
      - Strategy description (rules)
      - HTF POI details
      - LTF confirmation details
      - Potential targets
      - HTF candle table

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
        "Provide your reasoning and a clear decision: TRADE or NO TRADE.\n"
        "If TRADE, suggest an entry price, stop loss, and take profit level."
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
